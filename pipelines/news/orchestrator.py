"""The news-processing orchestrator. A synchronous driver that runs the agnostic nlp modules for
one article and owns all state (Weaviate I/O, the MinHash check, merge, taxonomy + scope config).
The modules stay stateless; this is the only place that knows 'the news use case'."""
import dataclasses
import json
import os

import httpx

from nlp import nli
from nlp.summarizer import service as summarizer
from nlp.geotagger import service as geotagger
from nlp.dedup import minhash, service as dedup_service

from . import config as cfg_mod, weaviate_io, merge as merge_mod
from .places import dedupe_places as _dedupe_places

_VECTORIZER_URL = os.environ.get("VECTORIZER_URL", "http://vectorizer:8000/embed")
_VECTORIZER_TIMEOUT = float(os.environ.get("VECTORIZER_TIMEOUT", "30"))


def run_article(article: dict) -> dict:
    """article: {title, text, source, url, date}. Returns a result dict. Performs exactly one
    state-changing operation — a merge (step 1/5) or the terminal upsert (step 10)."""
    cfg = cfg_mod.load()
    wv = weaviate_io.Client(cfg.collection)
    stage = "minhash"
    try:
        # 1. MinHash duplicate check
        sig = minhash.compute(article["text"])
        dup = _minhash_match(wv, sig)
        if dup:
            merge_mod.merge_article(wv, dup, article)
            return {"status": "duplicate", "stage": "minhash", "merged_into": dup}

        # 2. Summarize
        stage = "summarize"
        s = summarizer.summarize("article", {"title": article["title"], "text": article["text"]})
        head_sum = f"{s['headline']}\n{s['summary']}"

        # 3. Scope gate
        stage = "scope"
        if not _in_scope(head_sum, cfg):
            return {"status": "out_of_scope"}

        # 4. Embed
        stage = "embed"
        vec = _embed(head_sum)

        # 5. Embedding dedup
        stage = "dedup"
        item = dedup_service.dedup_item(cfg.collection, vec, kind="article",
                                        compare_text=head_sum, compare_property="summary")
        if item["decision"] == "match":
            merge_mod.merge_article(wv, item["target_id"], article)
            return {"status": "duplicate", "stage": "embedding", "merged_into": item["target_id"]}

        # 6-7. Geotag + deterministic place dedup
        stage = "geotag"
        places = geotagger.run(article["text"], headline=s["headline"])["places"]
        places = _dedupe_places(places)

        # 8. Topics
        stage = "topics"
        topics = _assign_topics(head_sum, cfg)

        # 9. Scope (nested)
        stage = "scope_assign"
        scope, region, city_ids = _assign_scope(head_sum, places, cfg)

        # 10. Upsert — the single terminal write
        stage = "upsert"
        obj_id = wv.upsert({
            "headline": s["headline"], "summary": s["summary"],
            "sources": [article["source"]], "urls": [article["url"]], "date": article["date"],
            "minhash": sig, "topics": topics, "scope": scope, "region": region,
            "city_ids": city_ids, "places": json.dumps(places, default=_asdict),
        }, vector=vec)
        return {"status": "indexed", "id": obj_id, "topics": topics, "scope": scope}

    except (httpx.HTTPError, RuntimeError):
        return {"status": "error", "stage": stage}


def _asdict(obj):
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    raise TypeError(f"not serializable: {type(obj)!r}")


def _embed(text: str) -> list[float]:
    r = httpx.post(_VECTORIZER_URL, json={"text": text}, timeout=_VECTORIZER_TIMEOUT)
    r.raise_for_status()
    return r.json()["embedding"]


def _minhash_match(wv, sig, threshold=None) -> str | None:
    """Brute-force Jaccard over a recent candidate window stored in Weaviate. Fine for catching
    near-identical re-publications at this corpus size; swap iter_recent for LSH bands if it grows."""
    threshold = (threshold if threshold is not None
                 else float(os.environ.get("NEWS_MINHASH_THRESHOLD", "0.9")))
    window = int(os.environ.get("NEWS_MINHASH_WINDOW", "5000"))
    for obj in wv.iter_recent(return_props=("minhash",), limit=window):
        if minhash.jaccard(sig, obj.properties["minhash"]) >= threshold:
            return str(obj.uuid)
    return None


def _in_scope(text, cfg) -> bool:
    # whitelist: OR — ANY in-scope hypothesis clears its threshold
    wl = nli.score(text, cfg.in_scope_hyps)
    if not any(p["score"] >= cfg.in_scope_threshold for p in wl):
        return False
    # blacklist: AND-exclusion — none may clear its threshold
    bl = nli.score(text, cfg.blacklist_hyps, threshold=cfg.blacklist_threshold, blacklist=True)
    if any(p["score"] >= cfg.blacklist_threshold for p in bl):
        return False
    return True


def _assign_topics(text, cfg) -> list[str]:
    scored = nli.score(text, cfg.topic_labels, hypothesis_template=cfg.topic_template)
    keep = sorted((p for p in scored if p["score"] >= cfg.topic_threshold),
                  key=lambda p: p["score"], reverse=True)[:cfg.topic_top_k]
    return [p["hypothesis"] for p in keep]


def _assign_scope(text, places, cfg):
    # Pass 1 — city / regional / national (mutually exclusive)
    cities = [p for p in places if p.type == "city"]
    regions = [p for p in places if p.type == "region"]
    geo_ctx = ""
    if cities:
        geo_ctx += " Ciudades: " + ", ".join(c.city_name for c in cities[:5]) + "."
    levels = list(cfg.scope_levels)                       # ["city","regional","national"]
    res = nli.classify(text + geo_ctx, labels=[cfg.scope_levels[l] for l in levels],
                       multi_label=False)
    best = res["labels"][0]
    scope = next(l for l in levels if cfg.scope_levels[l] == best)
    if res["scores"][0] < cfg.scope_threshold:            # weak → fall back to geometry
        scope = "city" if cities else ("regional" if regions else "national")

    # Pass 2 — which region / city (only when not national)
    region, city_ids = None, [c.city_id for c in cities if c.city_id is not None]
    if scope == "regional" and regions:
        r = nli.classify(text, labels=[x.name for x in regions],
                         multi_label=False, hypothesis_template=cfg.place_template)
        region = r["labels"][0]
    elif scope == "city" and len(cities) > 1:
        c = nli.classify(text, labels=[x.city_name for x in cities],
                         multi_label=False, hypothesis_template=cfg.place_template)
        winner = next(x for x in cities if x.city_name == c["labels"][0])
        city_ids = [winner.city_id] if winner.city_id is not None else city_ids
    return scope, region, city_ids
