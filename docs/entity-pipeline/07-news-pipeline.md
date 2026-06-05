# News Pipeline — Orchestrator Implementation Doc

**Scope:** the news-processing orchestrator · **Status:** New (written from scratch)
**Authoritative source:** `docs/superpowers/specs/2026-06-05-nlp-two-pipelines-design.md` §4

## Purpose

A **synchronous Python orchestrator** that drives the agnostic modules for one article, owns all
state (Weaviate I/O, the MinHash check, `merge`, taxonomy + scope config), and decides flow. The
modules stay stateless; this process is the only place that knows "the news use case".

It replaces the old per-endpoint ingestion driver (`ingestion/07_news/070_ingest_news.py`) and
the orchestration that used to live in `nlp/classifier/service.py` and the geotagger's scope code.

```
article {title, text, source, url, date}
  │
  1. minhash_check(text)               ── duplicate? ──► merge() ──► DONE
  │
  2. summarizer.summarize("article", {title, text})        → headline, summary
  │
  3. scope_gate(headline+summary)      ── out of scope? ──► DONE
  │     whitelist via nli.score (OR)   +  blacklist via nli.score (AND-exclusion)
  │
  4. vectorizer.embed(headline+summary)                    → vec (bge-m3)
  │
  5. dedup.dedup_item(ARTICLES, vec)   ── match? ──► merge() ──► DONE
  │
  6. geotagger.run(text, headline, source)                 → places[]
  │
  7. dedupe_places(places)                                 → places[] (deterministic)
  │
  8. assign_topics(headline+summary)   (nli.score, no threshold, keep ≥ topic_threshold)
  │
  9. assign_scope(headline+summary, places)  (nested nli.classify, multi_label=False)
  │
  10. upsert_article(...)              → Weaviate (record + minhash + vec + topics + scope + places)
```

---

## File layout

```
pipelines/
  news/
    __init__.py
    orchestrator.py     ← run_article(article): the 10-step flow above
    config.py           ← NewsConfig: loads topics.yaml + thresholds + collection name
    weaviate_io.py      ← collection ensure/create, minhash candidate fetch, upsert, merge
    merge.py            ← merge_article(): append source+url, keep oldest date
config/
  topics.yaml           ← in-scope / blacklist / topic / scope hypotheses + thresholds
```

The orchestrator imports the modules in-process: `from nlp import nli`,
`from nlp.summarizer import service as summarizer`, `from nlp.geotagger import service as
geotagger`, `from nlp.dedup import minhash, service as dedup_service`. The `vectorizer` is the
existing external service, called over HTTP.

---

## Configuration — `config/topics.yaml` + `NewsConfig`

All the taxonomy/threshold knowledge that the old `classifier` baked in now lives here, adapted to
the `nli.score` contract (doc 02).

```yaml
# config/topics.yaml
in_scope:                         # whitelist hypotheses — pass if ANY clears its threshold (OR)
  threshold: 0.40
  hypotheses:
    - "Este artículo trata sobre urbanismo, obras o movilidad en una ciudad."
    - "Este artículo trata sobre política o servicios municipales."
    # ...
blacklist:                        # exclude if ANY clears its threshold (AND-exclusion)
  threshold: 0.70
  hypotheses:
    - "Este artículo trata principalmente sobre deportes."
    - "Este artículo es una esquela o anuncio publicitario."
topics:                           # labels assigned to in-scope articles
  threshold: 0.50
  top_k: 3
  labels:
    - "movilidad"
    - "vivienda"
    # ...
  hypothesis_template: "Este texto trata sobre {}."
scope:                            # nested geographic scope
  threshold: 0.35
  level_hypotheses:               # pass 1 — mutually exclusive (nli.classify multi_label=False)
    city:     "Este artículo describe actuaciones de un ayuntamiento o municipio concreto."
    regional: "Este artículo describe políticas de una comunidad autónoma o provincia."
    national: "Este artículo describe un asunto de alcance estatal."
  place_template: "Este artículo trata principalmente sobre {}."   # pass 2 over detected places
```

```python
# pipelines/news/config.py
from dataclasses import dataclass
import os, yaml

@dataclass(frozen=True)
class NewsConfig:
    in_scope_hyps: list[str]; in_scope_threshold: float
    blacklist_hyps: list[str]; blacklist_threshold: float
    topic_labels: list[str]; topic_threshold: float; topic_top_k: int; topic_template: str
    scope_levels: dict[str, str]; scope_threshold: float; place_template: str
    collection: str = os.environ.get("NEWS_COLLECTION", "Article_news_v1")

def load(path: str | None = None) -> NewsConfig:
    data = yaml.safe_load(open(path or os.environ.get("TOPICS_YAML_PATH", "config/topics.yaml")))
    return NewsConfig(
        in_scope_hyps=data["in_scope"]["hypotheses"], in_scope_threshold=float(data["in_scope"]["threshold"]),
        blacklist_hyps=data["blacklist"]["hypotheses"], blacklist_threshold=float(data["blacklist"]["threshold"]),
        topic_labels=data["topics"]["labels"], topic_threshold=float(data["topics"]["threshold"]),
        topic_top_k=int(data["topics"]["top_k"]), topic_template=data["topics"]["hypothesis_template"],
        scope_levels=data["scope"]["level_hypotheses"], scope_threshold=float(data["scope"]["threshold"]),
        place_template=data["scope"]["place_template"],
    )
```

---

## Weaviate article collection (owned here)

The orchestrator creates the collection if absent (`weaviate_io.ensure_collection`). Vectors are
supplied (bge-m3); Weaviate does not vectorize.

```jsonc
{
  "name": "Article_news_v1",
  "vectorizer_config": "none",
  "properties": [
    {"name": "headline",  "dataType": ["text"]},
    {"name": "summary",   "dataType": ["text"]},
    {"name": "sources",   "dataType": ["text[]"]},     // appended on merge
    {"name": "urls",      "dataType": ["text[]"]},      // appended on merge
    {"name": "date",      "dataType": ["date"]},        // oldest kept on merge
    {"name": "minhash",   "dataType": ["int[]"]},       // dedup.minhash.compute(text)
    {"name": "topics",    "dataType": ["text[]"]},
    {"name": "scope",     "dataType": ["text"]},         // city | regional | national
    {"name": "region",    "dataType": ["text"]},
    {"name": "city_ids",  "dataType": ["int[]"]},
    {"name": "places",    "dataType": ["text"]}          // JSON-encoded places[] for provenance
  ],
  "vectorIndexConfig": {"distance": "cosine"}
}
```

---

## Orchestrator (`pipelines/news/orchestrator.py`)

```python
import json, os, httpx
from nlp import nli
from nlp.summarizer import service as summarizer
from nlp.geotagger import service as geotagger
from nlp.dedup import minhash, service as dedup_service
from . import config as cfg_mod, weaviate_io, merge as merge_mod

_VECTORIZER_URL = os.environ.get("VECTORIZER_URL", "http://vectorizer:8000/embed")


def run_article(article: dict) -> dict:
    """article: {article_id, title, text, source, url, date}. Returns a result dict."""
    cfg = cfg_mod.load()
    wv = weaviate_io.Client(cfg.collection)

    # 1. MinHash duplicate check (orchestrator-owned, against Weaviate)
    sig = minhash.compute(article["text"])
    dup = _minhash_match(wv, sig)
    if dup:
        merge_mod.merge_article(wv, dup, article)
        return {"status": "duplicate", "stage": "minhash", "merged_into": dup}

    # 2. Summarize
    s = summarizer.summarize("article", {"title": article["title"], "text": article["text"]})
    head_sum = f"{s['headline']}\n{s['summary']}"

    # 3. Scope gate (whitelist OR + blacklist AND-exclusion)
    if not _in_scope(head_sum, cfg):
        return {"status": "out_of_scope"}

    # 4. Embed
    vec = _embed(head_sum)

    # 5. Embedding dedup (2-step)
    item = dedup_service.dedup_item(cfg.collection, vec, kind="article",
                                    compare_text=head_sum, compare_property="summary")
    if item["decision"] == "match":
        merge_mod.merge_article(wv, item["target_id"], article)
        return {"status": "duplicate", "stage": "embedding", "merged_into": item["target_id"]}

    # 6. Geotag
    places = geotagger.run(article["text"], headline=s["headline"], source=article["source"])["places"]

    # 7. Dedupe places (deterministic — they already carry resolved ids)
    places = _dedupe_places(places)

    # 8. Topics
    topics = _assign_topics(head_sum, cfg)

    # 9. Scope (nested)
    scope, region, city_ids = _assign_scope(head_sum, places, article["source"], cfg)

    # 10. Upsert
    obj_id = wv.upsert({
        "headline": s["headline"], "summary": s["summary"],
        "sources": [article["source"]], "urls": [article["url"]], "date": article["date"],
        "minhash": sig, "topics": topics, "scope": scope, "region": region,
        "city_ids": city_ids, "places": json.dumps(places, default=_asdict),
    }, vector=vec)
    return {"status": "indexed", "id": obj_id, "topics": topics, "scope": scope}
```

### MinHash check (`_minhash_match`)

```python
def _minhash_match(wv, sig, threshold=None) -> str | None:
    """Brute-force Jaccard over recent candidate signatures stored in Weaviate.

    Weaviate has no native LSH; for 'easy duplicate' catching we fetch a candidate window
    (e.g. recent articles) and compare. Returns the matched object id or None.
    """
    threshold = threshold if threshold is not None else float(os.environ.get("NEWS_MINHASH_THRESHOLD", "0.9"))
    for obj in wv.iter_recent(return_props=("minhash",), limit=int(os.environ.get("NEWS_MINHASH_WINDOW", "5000"))):
        if minhash.jaccard(sig, obj.properties["minhash"]) >= threshold:
            return str(obj.uuid)
    return None
```

> **Scalability note:** brute-force over a recent window is fine for catching near-identical
> re-publications at this corpus size. If the article store grows large, replace `iter_recent`
> with an LSH-band bucket property (store band hashes, filter by matching band) — an internal
> change to `weaviate_io`, no contract change.

### Scope gate (`_in_scope`) — relocated from `classifier`

```python
def _in_scope(text, cfg) -> bool:
    # whitelist: OR — ANY in-scope hypothesis clears its threshold
    wl = nli.score(text, cfg.in_scope_hyps)                       # no threshold → all scores
    if not any(p["score"] >= cfg.in_scope_threshold for p in wl):
        return False
    # blacklist: AND-exclusion — none may clear its threshold
    bl = nli.score(text, cfg.blacklist_hyps, threshold=cfg.blacklist_threshold, blacklist=True)
    if any(p["score"] >= cfg.blacklist_threshold for p in bl):
        return False
    return True
```

### Topic assignment (`_assign_topics`)

```python
def _assign_topics(text, cfg) -> list[str]:
    scored = nli.score(text, cfg.topic_labels, hypothesis_template=cfg.topic_template)  # all
    keep = sorted((p for p in scored if p["score"] >= cfg.topic_threshold),
                  key=lambda p: p["score"], reverse=True)[:cfg.topic_top_k]
    return [p["hypothesis"] for p in keep]
```

### Scope assignment (`_assign_scope`) — relocated from the geotagger + classifier

```python
def _assign_scope(text, places, source, cfg):
    # Pass 1 — city / regional / national (mutually exclusive)
    cities  = [p for p in places if p.type == "city"]
    regions = [p for p in places if p.type == "region"]
    geo_ctx = ""
    if cities:  geo_ctx += " Ciudades: " + ", ".join(c.city_name for c in cities[:5]) + "."
    levels = list(cfg.scope_levels)                                 # ["city","regional","national"]
    res = nli.classify(text + geo_ctx, labels=[cfg.scope_levels[l] for l in levels],
                       multi_label=False)
    best = res["labels"][0]
    scope = next(l for l in levels if cfg.scope_levels[l] == best)
    if res["scores"][0] < cfg.scope_threshold:                      # weak → fall back to geometry
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
```

### Place dedup (`_dedupe_places`)

```python
def _place_key(p):
    """Identity key per type. Not every type carries the same id, and any place can be
    unresolved (no id) — so fall back to the normalized surface text, which keeps two distinct
    unresolved places from collapsing into one."""
    t = p.type
    if t == "region":
        return ("region", p.geonames_id or _norm(p.name or p.text))
    if t == "city":
        return ("city", p.city_id or p.geonames_id or _norm(p.city_name or p.text))
    if t == "street":
        # a street is its geometry (edge_ids); unresolved → (imputed city, surface text)
        return ("street", tuple(sorted(p.edge_ids)) if p.edge_ids
                else (p.city_id, _norm(p.text)))
    # location: gazetteer point id, else rounded coords, else surface text
    return ("location", p.geonames_id
            or ((round(p.lat, 4), round(p.lon, 4)) if p.lat is not None else _norm(p.text)))


def _dedupe_places(places) -> list:
    """Collapse repeated mentions deterministically (the resolver's pre-merge idea, no LLM)."""
    seen, out = set(), []
    for p in places:
        k = _place_key(p)
        if k in seen:
            continue
        seen.add(k); out.append(p)
    return out
```

> **Design note (flagged + corrected):** your step 7 said "call the normalizer to remove
> duplicates". The `resolver` (doc 05) merges *free-text mentions* with an LLM refine; the
> geotagger's places are already resolved, so a deterministic identity-key collapse is sufficient
> and avoids an unnecessary LLM call. **But not every type carries an id, and any place can be
> unresolved**, so the key is per-type with a normalized-text fallback — otherwise distinct
> unresolved places (e.g. two streets the b4c API didn't find) would wrongly merge. `region`
> covers country-level admin too (geonames feature_class A); split it by `feature_code` only if
> you later need a separate `country` type.

---

## `merge` (`pipelines/news/merge.py`)

```python
def merge_article(wv, target_id: str, article: dict) -> None:
    """Append the new source + url, keep the oldest date. No re-embedding, no re-summarizing."""
    cur = wv.get(target_id, return_props=("sources", "urls", "date"))
    sources = list(dict.fromkeys([*cur["sources"], article["source"]]))   # de-dup, preserve order
    urls    = list(dict.fromkeys([*cur["urls"], article["url"]]))
    oldest  = min(cur["date"], article["date"])
    wv.update(target_id, {"sources": sources, "urls": urls, "date": oldest})
```

`merge` is the orchestrator operation referenced across the spec — it lives here, not in any
module. It runs on both the MinHash hit (step 1) and the embedding-dedup match (step 5).

---

## Error handling

| Failure | Behaviour |
|---|---|
| `summarizer`/`nli` Ollama down (`httpx.HTTPError`) | Abort this article, return `{"status":"error","stage":...}`; the caller retries. No partial upsert. |
| `vectorizer` down | Same — abort before any write. |
| `geotagger` b4c API down | Abort (places are required for scope). Could be relaxed to "index without places" later. |
| Weaviate down | Abort; nothing is half-written (upsert is the single terminal write). |

The orchestrator is the only stateful actor, and it performs exactly one terminal write
(`upsert`) or one `merge` per article, so there is never a partially-ingested article.

---

## Configuration (env)

| Env var | Default | Description |
|---|---|---|
| `TOPICS_YAML_PATH` | `config/topics.yaml` | Taxonomy + scope hypotheses |
| `NEWS_COLLECTION` | `Article_news_v1` | Weaviate article collection |
| `NEWS_MINHASH_THRESHOLD` | `0.9` | Jaccard cut for MinHash duplicate |
| `NEWS_MINHASH_WINDOW` | `5000` | Candidate signatures scanned per check |
| `VECTORIZER_URL` | `http://vectorizer:8000/embed` | bge-m3 embedding service |
| `WEAVIATE_HTTP_HOST` / `WEAVIATE_GRPC_HOST` | `weaviate` | Weaviate (shared with dedup, doc 06) |

Module env vars (Ollama model/timeouts, `NLI_MODEL`, `B4C_API_BASE`, dedup thresholds) are owned
by their modules (docs 01–06).

---

## Dependencies

- Modules: `nlp.summarizer`, `nlp.nli`, `nlp.geotagger`, `nlp.dedup` (docs 01–06).
- `weaviate-client>=4` (shared with dedup), `httpx`, `pyyaml`.
- External `vectorizer` service (bge-m3) and Ollama sidecar.
- No Neo4j (that is pipeline 2 only).
