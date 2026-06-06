# Two-Pipeline Deployment Plan

**Scope:** everything that must be wired in to run the **news pipeline** (`pipelines/news`) and the
**relation pipeline** (`pipelines/relations`) in production. Date: 2026-06-06.

The two pipelines are synchronous Python **libraries** (`run_article`, `ingest_document`,
`consolidate`). They drive the agnostic `nlp` modules in-process and own all state. They are **not
yet exposed** by anything — the current `nlp-service` FastAPI app only serves the module endpoints
(`/summarize`, `/geotag`, `/nli`, `/ner`, `/resolve`, `/dedup`). Step 3 below closes that gap.

---

## 0. Dependency map (what each pipeline talks to)

| Dependency | News | Relations | Protocol | Notes |
|---|---|---|---|---|
| Ollama (LLM sidecar) | ✓ summarizer | ✓ ner, coref, resolver, summarizer | HTTP | models below |
| Weaviate | ✓ article collection | ✓ entity/relation working + final collections | HTTP REST+GraphQL (httpx) | vectors supplied; `vectorizer: none` |
| Vectorizer (bge-m3, 1024-d) | ✓ | ✓ | HTTP `POST {text}` → `{embedding}` | **contract must be verified — see step 2.4** |
| b4c cities API | ✓ geotagger | ✗ | HTTP | `B4C_API_BASE` (geotagger-owned) |
| Neo4j | ✗ | ✓ Phase B only | Bolt (neo4j driver) | **new service** |

Collections are created lazily by the orchestrators (`ensure_collection` / `ensure_collections`),
so there is no manual schema step — but the backing services must be reachable before first call.

---

## 1. Provision infrastructure (on `b4c-net`)

1.1 **Weaviate** — reachable at `WEAVIATE_URL` (default `http://weaviate:8080`). Anonymous access
or wire auth into `pipelines/*/weaviate_io.py` (currently no auth header). Persistent volume.

1.2 **Vectorizer (bge-m3)** — reachable at `VECTORIZER_URL` (default
`http://vectorizer:8000/embed`). Must return a 1024-d embedding. **This service is not in the repo
compose** — confirm it is deployed on `b4c-net`.

1.3 **Ollama** — reachable at `OLLAMA_HOST`. Pull the models referenced by the modules:
   - `OLLAMA_MODEL` (summarizer + coref), default `gemma4:31b`
   - `EXTRACTION_MODEL` (ner), default `gemma4:31b`
   - `RESOLVE_MODEL` (resolver), default `gemma4:31b`
   - the dedup LLM adjudicator model (see `nlp/dedup/llm.py` env)
   Run `ollama pull <model>` for each on the Ollama host.

1.4 **Neo4j** (relations only) — deploy a Neo4j 5.x instance on `b4c-net`. Add to the production
compose, e.g.:
```yaml
  neo4j:
    image: neo4j:5.24
    environment:
      NEO4J_AUTH: neo4j/${NEO4J_PASSWORD}
    ports: ["7474:7474", "7687:7687"]
    volumes: [neo4j_data:/data]
    networks: [b4c-net]
```

1.5 **b4c cities API** (news only) — already used by the geotagger; ensure `B4C_API_BASE` resolves
from the service container.

---

## 2. Configuration / environment

2.1 **Shared env** (add to the `nlp-service` environment block in compose):

| Env var | Value | Used by |
|---|---|---|
| `WEAVIATE_URL` | `http://weaviate:8080` | both |
| `VECTORIZER_URL` | `http://vectorizer:8000/embed` | both |
| `OLLAMA_HOST` / `OLLAMA_MODEL` / `OLLAMA_TIMEOUT` | existing | both |

2.2 **News pipeline env:**

| Env var | Default | Notes |
|---|---|---|
| `TOPICS_YAML_PATH` | `config/topics.yaml` | shipped in image |
| `NEWS_COLLECTION` | `Article_news_v1` | |
| `NEWS_MINHASH_THRESHOLD` | `0.9` | |
| `NEWS_MINHASH_WINDOW` | `5000` | candidate window scanned per article |

2.3 **Relation pipeline env:**

| Env var | Default | Notes |
|---|---|---|
| `ONTOLOGY_DIR` | `config/ontologies` | shipped in image |
| `RELATIONS_CHUNK_TOKENS` | `512` | words-as-proxy chunk budget |
| `EXTRACTION_MODEL` / `RESOLVE_MODEL` | `gemma4:31b` | module-owned |
| `NEO4J_URI` | `bolt://neo4j:7687` | |
| `NEO4J_USER` / `NEO4J_PASSWORD` | — | **required for Phase B** |

2.4 **Verify the vectorizer contract.** `pipelines/*/orchestrator.py::_embed` posts
`{"text": ...}` and reads `resp.json()["embedding"]`. If the real bge-m3 service differs (e.g.
`{"texts": [...]}` → `{"embeddings": [[...]]}`), adapt `_embed` in **both** orchestrators (or
extract a shared `pipelines/_vectorizer.py`). This is the single most likely integration break.

---

## 3. Wire an invocation surface (the missing entrypoint)

The pipelines are libraries; choose how they are driven. Recommended: **FastAPI routers**,
consistent with `api/routers/*`.

3.1 Add `api/routers/pipelines.py`:
```python
from fastapi import APIRouter
from pipelines.news import run_article
from pipelines.relations import ingest_document, consolidate

router = APIRouter(prefix="/pipelines", tags=["pipelines"])

@router.post("/news")
def news(article: dict):              return run_article(article)

@router.post("/relations/ingest")
def ingest(body: dict):               return ingest_document(body["doc"], body["use_case"])

@router.post("/relations/consolidate")
def cons(body: dict):                 return consolidate(body["use_case"])
```
Register it in `api/main.py::_register_routers`. (Define Pydantic request models instead of raw
`dict` for production validation.)

3.2 **Phase B is batch.** Do **not** call `/relations/consolidate` per document. Run it on a
schedule or after an ingest batch (cron / job). It is idempotent and re-runnable.

3.3 Alternative surfaces (pick per ops model): a CLI under `scripts/` (`python -m pipelines.news
...`) for batch backfills, or a queue worker consuming an ingest topic. The library API is the
same either way.

---

## 4. Build the image

4.1 `neo4j==5.24.0` is already in `requirements.txt` — it installs into the image.

4.2 Extend the Dockerfile build guard (currently asserts `config/topics.yaml`) to also assert the
ontologies are present:
```dockerfile
RUN test -f /app/config/topics.yaml && \
    test -f /app/config/ontologies/financial_flows.yaml
```

4.3 No new model pre-pull is needed in the image: flair NER + XNLI are already pre-pulled; the
LLM models live on the Ollama host (step 1.3), and bge-m3 lives in the vectorizer service.

4.4 `config/ontologies/` ships via the existing `COPY . /app/`. Add a new ontology YAML per use
case here; the collection name embeds `use_case` + `version`, so a schema change = bump `version:`
in the YAML (new collections, no migration of old ones).

---

## 5. Readiness / health

5.1 The existing `/readyz` only checks module warmth. The pipelines additionally need Weaviate,
the vectorizer, and (relations) Neo4j reachable. Optionally extend `/readyz` to ping those before
serving `/pipelines/*`.

5.2 Smoke after deploy:
```bash
curl -s $NLP/healthz
curl -s -XPOST $NLP/pipelines/news -d '{"title":"t","text":"...","source":"s","url":"u","date":"2026-06-06"}'
curl -s -XPOST $NLP/pipelines/relations/ingest \
     -d '{"use_case":"financial_flows","doc":{"doc_id":"d1","text":"..."}}'
curl -s -XPOST $NLP/pipelines/relations/consolidate -d '{"use_case":"financial_flows"}'
```
Expect `{"status":"indexed"|"duplicate"|"out_of_scope"}` for news, `{"doc_id","entities",
"relations"}` for ingest, `{"entities","relations"}` for consolidate.

---

## 6. Per-pipeline runbook

**News** — one terminal write per article (`merge` or `upsert`); aborts before any write on a
dependency failure and returns `{"status":"error","stage":...}`; caller retries. Tune
`config/topics.yaml` thresholds against the eval notebooks (`eval/nli-news-*`, `eval/geotag-*`).

**Relations** — Phase A is idempotent on `doc_id` (re-run overwrites); a per-document failure
returns `{"status":"error","stage":...}`. Phase B rebuilds clusters from current collections, so a
failed consolidate (e.g. Neo4j down) is simply re-run to backfill the graph — the final Weaviate
upsert is independent of the Neo4j write.

---

## 7. Open items to confirm before go-live

1. **Vectorizer contract** (step 2.4) — the highest-risk assumption.
2. **Weaviate auth** — add an API-key/header to `weaviate_io` if the cluster requires it.
3. **Entrypoint choice** (step 3) — routers vs CLI vs worker; not yet implemented.
4. **Neo4j provisioning** (step 1.4) — new service + credentials.
5. **Consolidation scheduling** (step 3.2) — when/how often Phase B runs.
