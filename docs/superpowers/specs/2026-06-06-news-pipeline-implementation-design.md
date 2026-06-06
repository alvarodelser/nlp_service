# News Pipeline — Implementation Design

**Status:** Approved design, ready for TDD implementation. Date: 2026-06-06.
**Derives from:** `docs/superpowers/specs/2026-06-05-nlp-two-pipelines-design.md` §4 and
`docs/entity-pipeline/07-news-pipeline.md`. Where doc 07 disagrees with the actual module code,
this document records the reconciled truth and wins.

## Purpose

A self-contained, synchronous Python library — `pipelines/news/` — that drives the agnostic
`nlp.*` modules for one article. It owns all state (Weaviate I/O, the MinHash check, `merge`,
taxonomy + scope config) and decides flow. The modules stay stateless; this package is the only
place that knows "the news use case". `run_article(article)` is the single entry point. No new
HTTP endpoint — a caller (CLI / ingest job) imports the package.

## Flow (`run_article`)

```
article {article_id?, title, text, source, url, date}
  1. minhash_check(text)              ── duplicate? ──► merge() ──► {"status":"duplicate","stage":"minhash"}
  2. summarizer.summarize("article", {title, text})       → headline, summary
  3. scope_gate(headline+summary)     ── out of scope? ──► {"status":"out_of_scope"}
  4. vectorizer.embed(headline+summary)                   → vec (bge-m3)
  5. dedup.dedup_item(collection, vec) ── match? ──► merge() ──► {"status":"duplicate","stage":"embedding"}
  6. geotagger.run(text, headline=headline)               → places[] (GeoEntity)
  7. dedupe_places(places)                                → places[] (deterministic)
  8. assign_topics(headline+summary)   (nli.score, keep ≥ threshold, top_k)
  9. assign_scope(headline+summary, places)  (nested nli.classify, multi_label=False)
  10. upsert(...)                      → Weaviate (one terminal write) ──► {"status":"indexed"}
```

Exactly one state-changing operation runs per article: a `merge` (step 1 or 5) or the terminal
`upsert` (step 10). No partial writes.

## File layout

```
pipelines/
  __init__.py
  news/
    __init__.py        ← exports run_article
    config.py          ← NewsConfig dataclass + load()
    weaviate_io.py     ← Client: ensure_collection, iter_recent, get, upsert, update
    merge.py           ← merge_article()
    places.py          ← dedupe_places + _place_key
    orchestrator.py    ← run_article + _minhash_match, _in_scope, _assign_topics, _assign_scope, _embed
config/
  topics.yaml          ← rewritten to in_scope/blacklist/topics/scope schema (mobility content migrated)
tests/
  test_news_config.py, test_news_orchestrator.py, test_news_places.py,
  test_news_merge.py, test_news_scope.py
```

`places.py` is split out of the orchestrator (doc 07 inlines it) so the deterministic place-dedup
is independently testable and the orchestrator file stays focused.

In-process imports: `from nlp import nli`, `from nlp.summarizer import service as summarizer`,
`from nlp.geotagger import service as geotagger`,
`from nlp.dedup import minhash, service as dedup_service`. The vectorizer is the external HTTP
service.

## Reconciliations with doc 07 (verified against module code)

1. **Geotagger signature.** Actual: `geotagger.run(text, headline="", debug=False)` — there is
   **no `source` parameter**. Call `geotagger.run(article["text"], headline=s["headline"])`.
   `_assign_scope` also drops its unused `source` argument.
2. **Weaviate client.** `nlp.dedup.weaviate_index.WeaviateIndex` is deliberately read-only
   ("orchestrators own all writes") and is left untouched — `dedup_item` keeps using it. The
   pipeline builds its own **write-capable** `weaviate_io.Client` over `httpx`: REST
   (`/v1/objects`, `/v1/schema`) for ensure/get/upsert/update, GraphQL for `iter_recent`. It
   reuses the `WEAVIATE_URL` / `WEAVIATE_TIMEOUT` conventions.
3. **topics.yaml schema.** The live file uses an older schema (`labels`,
   `relevance_hypothesis`, `scope_hypotheses`). It is rewritten to doc 07's schema
   (`in_scope` / `blacklist` / `topics` / `scope`), porting the existing cycling/urban-mobility
   content so behaviour is preserved. `NewsConfig.load()` reads the new schema.
4. **Places JSON.** `GeoEntity` is a dataclass; serialize with
   `json.dumps(places, default=dataclasses.asdict)`.
5. Verified to match doc 07 as-is: `summarizer.summarize("article", {...})` → `headline`,
   `summary`; `nli.score` / `nli.classify`; `dedup_service.dedup_item(collection, vec,
   kind="article", compare_text=head_sum, compare_property="summary")`; `minhash.compute` /
   `minhash.jaccard`.

## NewsConfig (config.py)

Frozen dataclass loaded from `config/topics.yaml` (path via `TOPICS_YAML_PATH`):
`in_scope_hyps/threshold`, `blacklist_hyps/threshold`, `topic_labels/threshold/top_k/template`,
`scope_levels` (dict level→hypothesis), `scope_threshold`, `place_template`, `collection`
(`NEWS_COLLECTION`, default `Article_news_v1`).

## Gates and assignment (orchestrator.py)

- **`_in_scope(text, cfg)`** — whitelist OR (`nli.score` no threshold, any ≥ `in_scope_threshold`)
  then blacklist AND-exclusion (`nli.score` threshold + `blacklist=True`, none ≥ threshold).
- **`_assign_topics(text, cfg)`** — `nli.score` over `topic_labels` with `topic_template`, keep
  scores ≥ `topic_threshold` sorted desc, truncate to `topic_top_k`, return hypotheses.
- **`_assign_scope(text, places, cfg)`** — pass 1: `nli.classify` over the three level hypotheses
  (`multi_label=False`) with appended city context; weak best score (< `scope_threshold`) falls
  back to geometry (city if cities, else regional if regions, else national). Pass 2 (non-national):
  pick region / city via `nli.classify` with `place_template`. Returns `(scope, region, city_ids)`.
- **`_minhash_match(wv, sig)`** — brute-force Jaccard over `wv.iter_recent` window
  (`NEWS_MINHASH_WINDOW`, default 5000) at `NEWS_MINHASH_THRESHOLD` (default 0.9).

## Place dedup (places.py)

`dedupe_places(places)` collapses repeated mentions deterministically via per-type identity keys
with a normalized-text fallback (region→geonames_id; city→city_id/geonames_id; street→sorted
edge_ids or (city_id, text); location→geonames_id/rounded coords/text), exactly as doc 07
specifies. No LLM — the geotagger's places are already resolved.

## merge (merge.py)

`merge_article(wv, target_id, article)` — append source + url (de-duped, order-preserving), keep
the oldest date. No re-embedding, no re-summarizing.

## Weaviate collection (`Article_news_v1`)

`vectorizer_config: none` (vectors supplied). Properties: `headline`, `summary`, `sources[]`,
`urls[]`, `date`, `minhash int[]`, `topics[]`, `scope`, `region`, `city_ids int[]`,
`places` (JSON string). Cosine distance. The `summary` property + supplied vector are what
`dedup_item` reads back.

## Error handling

| Failure | Behaviour |
|---|---|
| `summarizer`/`nli` Ollama down (`httpx.HTTPError`) | Abort, `{"status":"error","stage":...}`; caller retries. No write. |
| `vectorizer` down | Same — abort before any write. |
| `geotagger` b4c API down | Abort (places required for scope). |
| Weaviate down | Abort; upsert is the single terminal write, so nothing is half-written. |

## Env config

`TOPICS_YAML_PATH` (`config/topics.yaml`), `NEWS_COLLECTION` (`Article_news_v1`),
`NEWS_MINHASH_THRESHOLD` (`0.9`), `NEWS_MINHASH_WINDOW` (`5000`),
`VECTORIZER_URL` (`http://vectorizer:8000/embed`), `WEAVIATE_URL` (`http://weaviate:8080`).
Module env vars (Ollama, `NLI_MODEL`, `B4C_API_BASE`, dedup thresholds) stay owned by their modules.

## Testing (TDD, mocked deps)

Tests written before implementation. Mock `nlp.summarizer/nli/geotagger/dedup`, the
`weaviate_io.Client`, and the vectorizer HTTP call — no live Ollama / Weaviate / b4c. Coverage:
config load; scope gate (in-scope, out-of-scope-by-whitelist, blacklisted); minhash dup; embedding
dup; topic keep / top_k; nested scope (city / regional / national, weak-score geometry fallback,
pass-2 region/city pick); place-dedup collapse across types incl. unresolved fallback; merge
(source/url append + oldest date).

## Dependencies

Modules `nlp.summarizer`, `nlp.nli`, `nlp.geotagger`, `nlp.dedup`; `httpx` + `pyyaml`; external
`vectorizer` + Ollama sidecar. No Neo4j (that is pipeline 2).
