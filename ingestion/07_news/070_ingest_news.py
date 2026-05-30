"""
Ingestion orchestrator — Phase 7 of run_ingestion.sh.

Reads new articles from movilidad_news_new.json, runs the 7-step NLP pipeline,
and writes one atomic DB record per article.

Pipeline order:
  1  POST /dedup-check       MinHash text fingerprint
  2  POST /extract           TF-IDF extract + 384-dim embedding
  3  POST /dedup-check-embed FAISS vector lookup (pre-computed embedding)
  4  local                   OOD relevance gate (cosine to centroid)
  5  POST /summarize         LLM headline + summary + summary embedding
  6  POST /geotag            NER + city/street resolution (no scope)
  6b local                   Edge lookup — resolve street spans to edge_ids via DB
  7  POST /classify          Topic NLI + scope NLI (joint fusion)
  DB  atomic INSERT          All 13 derived fields in one transaction
"""

import asyncio
import json
import logging
import os
import re
import time
import unicodedata
from pathlib import Path

import asyncpg
import httpx
import numpy as np

NLP_BASE = os.getenv("NLP_SERVICE_URL", "http://localhost:8000")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@db:5432/mobility")
OOD_CENTROID_PATH = os.getenv("OOD_CENTROID_PATH", "config/centroid.npy")
OOD_THRESHOLD = float(os.getenv("OOD_THRESHOLD", "0.75"))
SOURCE_PROFILE_PATH = os.getenv("SOURCE_PROFILE_PATH", "config/source_profile.json")
MAX_LLM_RETRIES = int(os.getenv("SUMMARIZE_MAX_RETRIES", "3"))
NEWS_NEW_PATH = os.getenv("NEWS_NEW_PATH", "data/news/movilidad_news_new.json")

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_centroid: np.ndarray | None = None
_source_profiles: dict = {}

# ── counters ──────────────────────────────────────────────────────────────────

_counts = {
    "text_duplicate": 0,
    "embed_duplicate": 0,
    "out_of_scope": 0,
    "llm_failure": 0,
    "complete": 0,
}
_step_times: dict[str, list[float]] = {k: [] for k in [
    "dedup_check", "extract", "dedup_embed", "relevance",
    "summarize", "geotag", "edge_lookup", "classify", "db_write",
]}


# ── startup ───────────────────────────────────────────────────────────────────

def _load_config() -> None:
    global _centroid, _source_profiles
    if Path(OOD_CENTROID_PATH).exists():
        _centroid = np.load(OOD_CENTROID_PATH).astype(np.float32)
    if Path(SOURCE_PROFILE_PATH).exists():
        _source_profiles = json.loads(Path(SOURCE_PROFILE_PATH).read_text())


def _source_profile(source: str) -> dict | None:
    return _source_profiles.get(source)


# ── street normalisation (mirrors nlp/geotagger/gazetteer.py) ─────────────────

_STREET_PREFIX_RE = re.compile(
    r"^(Calle|Avda?\.?|Avenida|Plaza|Paseo|Ronda|Travesía|Carretera|C/)\s+",
    flags=re.IGNORECASE,
)


def _normalise_street_span(span: str) -> str:
    stripped = _STREET_PREFIX_RE.sub("", span.strip())
    nfkd = unicodedata.normalize("NFKD", stripped.lower().strip())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


async def _resolve_street_edges(
    db: asyncpg.Connection,
    geo_streets: list[dict],
) -> list[dict]:
    """
    For each street entry returned by /geotag, query the edges table to fill
    in edge_ids for that city.  Entries already carrying edge_ids (from a
    pre-built street_index snapshot) are left untouched.
    """
    if not geo_streets:
        return geo_streets

    result = []
    for street in geo_streets:
        if street.get("edge_ids"):          # already resolved by NLP snapshot
            result.append(street)
            continue

        city_id = street.get("city_id")
        span    = street.get("span", "")
        if not city_id or not span:
            result.append(street)
            continue

        norm = _normalise_street_span(span)
        try:
            rows = await db.fetch(
                """
                SELECT id FROM edges
                WHERE city_id = $1
                  AND lower(unaccent(name)) = $2
                """,
                city_id, norm,
            )
            edge_ids = [row["id"] for row in rows]
        except Exception as exc:
            log.warning("Edge lookup failed '%s' city_id=%s: %s", span, city_id, exc)
            edge_ids = []

        result.append({**street, "edge_ids": edge_ids})

    return result


# ── pipeline ──────────────────────────────────────────────────────────────────

async def ingest_article(article: dict, db: asyncpg.Connection, client: httpx.AsyncClient) -> str:
    article_id = article["article_id"]
    raw_text = article["raw_text"]
    headline = article["headline"]
    source = article.get("source", "")
    url = article.get("url", "")
    pub_date = article.get("pub_date", "")
    search_tags = article.get("search_tags", [])
    source_profile = _source_profile(source)

    # ── Step 1: MinHash dedup ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    r = await client.post("/dedup-check", json={"article_id": article_id, "text": raw_text})
    r.raise_for_status()
    _step_times["dedup_check"].append(time.perf_counter() - t0)

    if (dup_id := r.json()["duplicate_of"]):
        await _enrich_record(db, dup_id, source, url, pub_date)
        _counts["text_duplicate"] += 1
        log.info("%s → text duplicate of %s", article_id, dup_id)
        return "text_duplicate"

    # ── Step 2: Extract + embed ───────────────────────────────────────────────
    t0 = time.perf_counter()
    r = await client.post("/extract", json={"article_id": article_id, "text": raw_text})
    r.raise_for_status()
    _step_times["extract"].append(time.perf_counter() - t0)

    extracted = r.json()
    extract_text = extracted["extract"]
    embedding_raw = np.array(extracted["embedding_raw"], dtype=np.float32)

    # ── Step 3: Embedding dedup ───────────────────────────────────────────────
    t0 = time.perf_counter()
    r = await client.post("/dedup-check-embed", json={
        "article_id": article_id, "embedding_raw": embedding_raw.tolist()
    })
    r.raise_for_status()
    _step_times["dedup_embed"].append(time.perf_counter() - t0)

    if (dup_id := r.json()["duplicate_of"]):
        await _enrich_record(db, dup_id, source, url, pub_date)
        _counts["embed_duplicate"] += 1
        log.info("%s → semantic duplicate of %s", article_id, dup_id)
        return "embed_duplicate"

    # ── Step 4: Relevance gate ────────────────────────────────────────────────
    t0 = time.perf_counter()
    in_scope = _check_relevance(embedding_raw)
    _step_times["relevance"].append(time.perf_counter() - t0)

    if not in_scope:
        await _insert_oos(db, article, embedding_raw)
        _counts["out_of_scope"] += 1
        log.info("%s → out of scope", article_id)
        return "out_of_scope"

    # ── Step 5: Summarize (with validation + retry) ───────────────────────────
    t0 = time.perf_counter()
    result = await _summarize_with_retry(client, article_id, raw_text, extract_text, headline)
    _step_times["summarize"].append(time.perf_counter() - t0)

    if result is None:
        _counts["llm_failure"] += 1
        log.warning("%s → LLM failed after %d retries", article_id, MAX_LLM_RETRIES)
        return "llm_failure"

    new_headline, summary, embedding_summary = result

    # ── Step 6: Geotag ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    r = await client.post("/geotag", json={
        "article_id": article_id, "text": raw_text, "headline": new_headline
    })
    r.raise_for_status()
    _step_times["geotag"].append(time.perf_counter() - t0)
    geo = r.json()

    # ── Step 6b: Resolve street spans to edge_ids ────────────────────────────
    t0 = time.perf_counter()
    geo["geo_streets"] = await _resolve_street_edges(db, geo["geo_streets"])
    _step_times["edge_lookup"].append(time.perf_counter() - t0)

    # ── Step 7: Topic + scope classification ──────────────────────────────────
    t0 = time.perf_counter()
    r = await client.post("/classify", json={
        "article_id": article_id,
        "summary": summary,
        "geo_cities": geo["geo_cities"],
        "search_tags": search_tags,
        "source_profile": source_profile,
    })
    r.raise_for_status()
    _step_times["classify"].append(time.perf_counter() - t0)
    classification = r.json()

    # ── Atomic DB write ───────────────────────────────────────────────────────
    t0 = time.perf_counter()
    await _insert_complete(db, {
        "article_id": article_id,
        "headline": new_headline,
        "summary": summary,
        "raw_text": raw_text,
        "source": source,
        "url": url,
        "pub_date": pub_date,
        "embedding_raw": embedding_raw.tolist(),
        "embedding_summary": embedding_summary,
        "sources": [{"name": source, "link": url, "date": pub_date}],
        "dates": [pub_date],
        "topics": classification["topics"],
        "scores": classification["scores"],
        "geo_scope": classification["geo_scope"],
        "geo_cities": geo["geo_cities"],
        "geo_streets": geo["geo_streets"],
    })
    _step_times["db_write"].append(time.perf_counter() - t0)

    _counts["complete"] += 1
    log.info("%s → complete  scope=%s  topics=%s", article_id,
             classification["geo_scope"], classification["topics"])
    return "complete"


# ── helpers ───────────────────────────────────────────────────────────────────

def _check_relevance(embedding: np.ndarray) -> bool:
    if _centroid is None:
        return True  # no centroid configured — pass all articles
    return float(np.dot(embedding, _centroid)) >= OOD_THRESHOLD


async def _summarize_with_retry(
    client: httpx.AsyncClient,
    article_id: str,
    text: str,
    extract: str,
    headline: str,
) -> tuple[str, str, list[float]] | None:
    for attempt in range(MAX_LLM_RETRIES):
        try:
            r = await client.post("/summarize", json={
                "article_id": article_id, "text": text, "extract": extract, "headline": headline
            })
            r.raise_for_status()
            result = r.json()
            if _validate_summary(result):
                return result["headline"], result["summary"], result["embedding_summary"]
            log.warning("%s summarize attempt %d failed validation", article_id, attempt + 1)
        except Exception as exc:
            log.warning("%s summarize attempt %d error: %s", article_id, attempt + 1, exc)
    return None


def _validate_summary(result: dict) -> bool:
    return (
        bool(result.get("headline"))
        and bool(result.get("summary"))
        and len(result["summary"].split()) >= 20
        and bool(result.get("embedding_summary"))
    )


async def _enrich_record(
    db: asyncpg.Connection, article_id: str, source: str, url: str, pub_date: str
) -> None:
    source_entry = json.dumps({"name": source, "link": url, "date": pub_date})
    await db.execute("""
        UPDATE news
        SET
          sources = CASE
            WHEN sources @> $2::jsonb THEN sources
            ELSE sources || $2::jsonb
          END,
          dates = CASE
            WHEN dates @> to_jsonb($3::text) THEN dates
            ELSE dates || to_jsonb($3::text)
          END
        WHERE article_id = $1
    """, article_id, f"[{source_entry}]", pub_date)


async def _insert_oos(db: asyncpg.Connection, article: dict, embedding_raw: np.ndarray) -> None:
    await db.execute("""
        INSERT INTO news (
          article_id, headline, raw_text, source, url, pub_date,
          embedding_raw, out_of_scope,
          sources, dates, topics, geo_cities, geo_streets
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,true,'[]','[]','[]','[]','[]')
        ON CONFLICT (article_id) DO NOTHING
    """,
        article["article_id"], article["headline"], article["raw_text"],
        article.get("source", ""), article.get("url", ""), article.get("pub_date", ""),
        embedding_raw.tolist(),
    )


async def _insert_complete(db: asyncpg.Connection, data: dict) -> None:
    await db.execute("""
        INSERT INTO news (
          article_id, headline, summary, raw_text, source, url, pub_date,
          embedding_raw, embedding_summary,
          sources, dates,
          topics, scores,
          geo_scope, geo_cities, geo_streets,
          out_of_scope
        ) VALUES (
          $1,$2,$3,$4,$5,$6,$7,
          $8,$9,
          $10::jsonb,$11::jsonb,
          $12::jsonb,$13::jsonb,
          $14,$15::jsonb,$16::jsonb,
          false
        )
        ON CONFLICT (article_id) DO NOTHING
    """,
        data["article_id"], data["headline"], data["summary"],
        data["raw_text"], data["source"], data["url"], data["pub_date"],
        data["embedding_raw"], data["embedding_summary"],
        json.dumps(data["sources"]), json.dumps(data["dates"]),
        json.dumps(data["topics"]), json.dumps(data["scores"]),
        data["geo_scope"],
        json.dumps(data["geo_cities"]), json.dumps(data["geo_streets"]),
    )


# ── main ──────────────────────────────────────────────────────────────────────

def _log_summary() -> None:
    log.info("── Pipeline summary ──────────────────────────")
    for key, count in _counts.items():
        log.info("  %-20s %d", key, count)
    log.info("── Step latencies (mean seconds) ─────────────")
    for step, times in _step_times.items():
        if times:
            log.info("  %-20s %.2f", step, sum(times) / len(times))


async def main() -> None:
    _load_config()

    articles = json.loads(Path(NEWS_NEW_PATH).read_text())
    log.info("Loaded %d articles from %s", len(articles), NEWS_NEW_PATH)

    db = await asyncpg.connect(DATABASE_URL)
    async with httpx.AsyncClient(base_url=NLP_BASE, timeout=180.0) as client:
        for article in articles:
            try:
                await ingest_article(article, db, client)
            except Exception as exc:
                log.error("Unhandled error for %s: %s", article.get("article_id"), exc)

    await db.close()
    _log_summary()


if __name__ == "__main__":
    asyncio.run(main())
