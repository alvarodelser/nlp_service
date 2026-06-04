# Dedup Module — Unified Weaviate + MinHash Rewrite

## Purpose of this document

Replaces the existing `nlp/dedup/` implementation with a unified, stateless-friendly
design and adds cross-document entity disambiguation as a second dedup path.

Both paths share **one API endpoint** (`POST /dedup`) parameterised by `type`. Both paths
follow the same pattern: pre-computed embedding in → Weaviate query → scored decision out.

Article dedup keeps a MinHash pre-filter (in-process, file-backed) to avoid Weaviate
calls for cheap exact/near-exact duplicates without incurring any model cost.

---

## What changes

| Component | Status |
|---|---|
| `minhash_index.py` | **Kept** — unchanged |
| `persistence.py` | **Simplified** — MinHash only; FAISS serialisation removed |
| `embedding_index.py` | **Deleted** — FAISS replaced by Weaviate |
| `id_map.py` | **Deleted** — row mapping no longer needed |
| `service.py` | **Rewritten** — stateless entity path; article path keeps MinHash + adds Weaviate stage 2 |
| `article_index.py` | **New** — Weaviate-backed article embedding search |
| `entity_index.py` | **New** — Weaviate-backed entity candidate search |
| `disambiguator.py` | **New** — three-tier entity decision logic |
| `api/routers/dedup.py` | **Unified** — single `POST /dedup` with `type` discriminator |
| `api/models.py` | **Updated** — discriminated-union request/response; existing article models replaced |

---

## File layout (after rewrite)

```
nlp/
  dedup/
    __init__.py
    minhash_index.py    ← unchanged
    persistence.py      ← MinHash only (FAISS serialisation removed)
    article_index.py    ← NEW: Weaviate-backed article embedding search
    entity_index.py     ← NEW: Weaviate-backed entity candidate search
    disambiguator.py    ← NEW: scoring + three-tier entity decision
    service.py          ← rewritten: check() + entity_disambiguate(), no global FAISS state

api/
  routers/
    dedup.py            ← POST /dedup (unified)
```

---

## Unified API

### `POST /dedup`

```python
# --- Requests ---

class ArticleDedupRequest(BaseModel):
    type:       Literal["article"] = "article"
    use_case:   str                          # pipeline identifier — indexes are namespaced by this
    article_id: str
    text:       str
    embedding:  list[float] | None = None   # 384-dim MiniLM; computed inline if absent

class EntityDedupRequest(BaseModel):
    type:         Literal["entity"] = "entity"
    use_case:     str
    local_entity: LocalEntity          # from /cluster output
    embedding:    list[float]          # 1024-dim bge-m3, caller-computed

DedupRequest = Annotated[
    ArticleDedupRequest | EntityDedupRequest,
    Field(discriminator="type")
]

# --- Responses ---

class ArticleDedupResponse(BaseModel):
    type:         Literal["article"] = "article"
    duplicate_of: str | None
    score:        float | None
    stage:        Literal["minhash", "embedding"] | None   # which stage caught the duplicate
    indexed:      bool

class EntityDedupResponse(BaseModel):
    type:         Literal["entity"] = "entity"
    local_id:     str
    decision:     Literal["merge", "create", "review"]
    canonical_id: str | None
    confidence:   float
    candidates:   list[CandidateResult]

DedupResponse = Annotated[
    ArticleDedupResponse | EntityDedupResponse,
    Field(discriminator="type")
]
```

Router:

```python
@router.post("/dedup")
def dedup(req: DedupRequest) -> DedupResponse:
    if req.type == "article":
        return dedup_service.check(req)
    return dedup_service.entity_disambiguate(req)
```

---

## Article dedup (two-stage)

### Stage 1 — MinHash (in-process, file-backed)

MinHash is a fast approximate Jaccard similarity check on word shingles. It requires no
neural model and no network call — just hashing. The implementation (`minhash_index.py`,
unchanged) uses **128 hash permutations** and **3-word shingles**, with a default threshold
of 0.9 Jaccard similarity. Any two articles with ≥ 90% overlapping 3-gram vocabulary are
treated as near-duplicates without ever touching Weaviate.

**Where it lives:** The `MinHashLSH` objects live in the NLP service process memory as a
dict keyed by `use_case`: `_mh: dict[str, MinHashIndex]`. Each pipeline gets its own
independent index — articles from `financial_flows` are never checked against articles from
another use case. They are loaded from pickle files on first use and flushed back to disk
every `DEDUP_PERSIST_EVERY_N` articles. The file for each use case is
`DEDUP_DATA_DIR/{use_case}/minhash_lsh.pkl` (e.g. `/data/dedup/financial_flows/minhash_lsh.pkl`).
This directory **must be a Docker volume mount** — without it the indexes reset to empty on
every container restart.

**Size:** Each article contributes one MinHash signature (128 × 4 bytes = 512 bytes) stored
in `_signatures`, plus entries in the LSH band tables (datasketch creates ≈ 25 bands for
threshold 0.9, each band a hash-table lookup entry). Total cost is roughly **3–5 KB per
article**. At 100K articles: ~300–500 MB RAM and a similar-sized pickle on disk. This is
fine for a single-process deployment. If you need to scale horizontally across multiple
replicas, each replica has independent state — the MinHash index does not synchronise.
For a single ingestion pipeline (one process ingesting articles sequentially) this is
not a problem.

**What is passed:** The article `text` must be sent to the service because MinHash operates
on word shingles of the raw text — the computation cannot be pre-delegated to the caller.
Stage 1 is handled entirely by the existing `minhash_index.py` logic, unchanged.

### Stage 2 — Weaviate embedding search

If MinHash misses, compute a 384-dim embedding (MiniLM-L12-v2) and query the
`Articles_{use_case}` Weaviate collection for the nearest article above
`DEDUP_ARTICLE_THRESHOLD`. On a non-duplicate decision, add the article to both the
MinHash index and Weaviate so future articles are checked against it.

### `nlp/dedup/article_index.py`

```python
class ArticleIndex:
    def __init__(self, use_case: str, weaviate_client) -> None:
        self._client = weaviate_client
        self._collection = f"Articles_{use_case}"   # one collection per pipeline

    def find_near_duplicate(
        self,
        embedding: list[float],
        threshold: float,
    ) -> tuple[str | None, float]:
        """Return (article_id, cosine_score) of nearest article, or (None, 0.0)."""
        results = (
            self._client.collections.get(self._collection)
            .query.near_vector(
                near_vector=embedding,
                limit=1,
                return_properties=["article_id"],
                return_metadata=MetadataQuery(distance=True),
            )
        )
        if not results.objects:
            return None, 0.0
        obj = results.objects[0]
        score = 1.0 - obj.metadata.distance
        return (obj.properties["article_id"], score) if score >= threshold else (None, 0.0)

    def add(self, article_id: str, embedding: list[float]) -> None:
        self._client.collections.get(self._collection).data.insert(
            properties={"article_id": article_id},
            vector=embedding,
        )
```

### `service.py` — article check

`_mh` and `_lock` are dicts keyed by `use_case`. `_load(use_case)` loads the pickle from
`DEDUP_DATA_DIR/{use_case}/minhash_lsh.pkl` on first call for that use case and is a
no-op on subsequent calls.

```python
_mh:    dict[str, MinHashIndex] = {}
_locks: dict[str, threading.Lock] = {}

def _load(use_case: str) -> None:
    if use_case not in _mh:
        _mh[use_case] = persistence.load_minhash(use_case)
        _locks[use_case] = threading.Lock()

def check(req: ArticleDedupRequest) -> ArticleDedupResponse:
    _load(req.use_case)
    with _locks[req.use_case]:
        mh = _mh[req.use_case]

        # Stage 1 — MinHash
        if req.article_id in mh:
            return ArticleDedupResponse(duplicate_of=req.article_id, score=1.0,
                                        stage="minhash", indexed=False)
        best_aid, jaccard = mh.query(req.text)
        if best_aid and jaccard >= mh.threshold:
            return ArticleDedupResponse(duplicate_of=best_aid, score=jaccard,
                                        stage="minhash", indexed=False)

        # Stage 2 — Weaviate
        vec = req.embedding or _encode(req.text)
        idx = ArticleIndex(req.use_case, _get_weaviate_client())
        dup_id, score = idx.find_near_duplicate(vec, _ARTICLE_THRESHOLD)
        if dup_id:
            return ArticleDedupResponse(duplicate_of=dup_id, score=score,
                                        stage="embedding", indexed=False)

        # Not a duplicate — index in both
        mh.add(req.article_id, req.text)
        idx.add(req.article_id, vec)
        _maybe_flush(req.use_case)
        return ArticleDedupResponse(duplicate_of=None, score=None,
                                    stage=None, indexed=True)
```

---

## Entity disambiguation (Weaviate only)

No MinHash stage. Entity names are too short and diverse for Jaccard to be a useful
pre-filter; Weaviate semantic search handles the full lookup.

### Decision logic

```
Query Weaviate for top-K candidates filtered by entity type
         │
         ▼
  Score each candidate:
    + alias bonus:    +0.15 if local entity name in candidate.aliases (capped at 1.0)
    - subtype penalty: -0.05 if subtypes differ
         │
         ▼
  best_score >= DISAMBIG_MERGE_THRESHOLD   →  "merge"
  DISAMBIG_REVIEW_LOW <= best_score < DISAMBIG_MERGE_THRESHOLD  →  LLM adjudication
  best_score < DISAMBIG_REVIEW_LOW         →  "create"
  no candidates                            →  "create"

  LLM adjudication:  "yes" → merge  |  "no" → create  |  "unsure" → review
```

### `nlp/dedup/entity_index.py`

```python
@dataclass
class Candidate:
    canonical_id:   str
    canonical_name: str
    type:           str
    subtype:        str | None
    description:    str
    aliases:        list[str]
    score:          float

class EntityIndex:
    def __init__(self, use_case: str, weaviate_client) -> None:
        self._client = weaviate_client
        self._collection = f"Entity_{use_case}_v1"

    def find_candidates(
        self, embedding: list[float], entity_type: str, top_k: int
    ) -> list[Candidate]:
        results = (
            self._client.collections.get(self._collection)
            .query.near_vector(
                near_vector=embedding,
                limit=top_k,
                filters=Filter.by_property("type").equal(entity_type),
                return_properties=["canonical_name","type","subtype","description","aliases"],
                return_metadata=MetadataQuery(distance=True),
            )
        )
        return [
            Candidate(
                canonical_id=str(o.uuid),
                canonical_name=o.properties["canonical_name"],
                type=o.properties["type"],
                subtype=o.properties.get("subtype"),
                description=o.properties.get("description", ""),
                aliases=o.properties.get("aliases", []),
                score=1.0 - o.metadata.distance,
            )
            for o in results.objects
        ]
```

### `nlp/dedup/disambiguator.py`

Pure logic — no external dependencies. Fully testable without Weaviate or Ollama.

```python
@dataclass
class DisambiguationResult:
    decision:     Literal["merge", "create", "review"]
    canonical_id: str | None
    confidence:   float
    candidates:   list[Candidate]

def disambiguate(
    local_entity:    LocalEntity,
    embedding:       list[float],
    entity_index:    EntityIndex,
    merge_threshold: float,
    review_low:      float,
    top_k:           int,
    llm_adjudicate:  bool = True,
) -> DisambiguationResult:
    candidates = entity_index.find_candidates(embedding, local_entity.type, top_k)
    if not candidates:
        return DisambiguationResult("create", None, 1.0, [])

    best = _apply_score_adjustments(candidates, local_entity)

    if best.score >= merge_threshold:
        return DisambiguationResult("merge", best.canonical_id, best.score, candidates)
    if best.score < review_low or not llm_adjudicate:
        return DisambiguationResult(
            "create" if best.score < review_low else "review",
            None, best.score, candidates,
        )
    verdict = _llm_adjudicate(local_entity, best)
    decision = {"yes": "merge", "no": "create"}.get(verdict, "review")
    canonical_id = best.canonical_id if decision == "merge" else None
    return DisambiguationResult(decision, canonical_id, best.score, candidates)
```

`_llm_adjudicate` uses `EXTRACTION_MODEL` with a 30-second timeout. On Ollama error
returns `"unsure"` (→ `"review"`) and logs WARNING.

---

## Weaviate client

```python
def _get_weaviate_client():
    import weaviate
    return weaviate.connect_to_custom(
        http_host=os.environ.get("WEAVIATE_HOST", "weaviate"),
        http_port=int(os.environ.get("WEAVIATE_PORT", "8080")),
        http_secure=False,
        grpc_host=os.environ.get("WEAVIATE_HOST", "weaviate"),
        grpc_port=int(os.environ.get("WEAVIATE_GRPC_PORT", "50051")),
        grpc_secure=False,
    )
```

Created per-request; SDK manages connection pool internally.

---

## Configuration

| Env var | Default | Description |
|---|---|---|
| `DEDUP_DATA_DIR` | `/data/dedup` | Root dir for MinHash pickle files; one subdir per use case |
| `DEDUP_PERSIST_EVERY_N` | `10` | Flush MinHash to file every N indexed articles |
| `DEDUP_ARTICLE_THRESHOLD` | `0.85` | Weaviate cosine threshold for article duplicates |
| `DISAMBIG_MERGE_THRESHOLD` | `0.92` | Entity: auto-merge above |
| `DISAMBIG_REVIEW_LOW` | `0.75` | Entity: auto-create below |
| `DISAMBIG_TOP_K` | `10` | Weaviate candidate count |
| `WEAVIATE_HOST` | `weaviate` | Weaviate hostname |
| `WEAVIATE_PORT` | `8080` | Weaviate HTTP port |
| `WEAVIATE_GRPC_PORT` | `50051` | Weaviate gRPC port |
| `ENTITY_EMBED_MODEL` | `bge-m3` | Ollama model for entity embeddings (caller-side) |

---

## Testing

See [06-testing.md](06-testing.md) §3.
- `disambiguator.py` is pure logic; candidates are injected — no Weaviate or Ollama needed.
- `article_index.py` and `entity_index.py` use an injected mock Weaviate client.
- `service.check()` is tested with mocked MinHash + mocked Weaviate: MinHash hit, Weaviate
  hit, full miss (indexed), Weaviate unavailable (503 propagated).
- Key entity cases: auto-merge, auto-create, each LLM verdict, alias bonus, subtype
  penalty, empty candidate list.

---

## Dependencies

- `weaviate-client>=4.0` — new package
- `sentence-transformers` — kept for inline article embedding fallback
- `faiss-cpu` — **removed** from requirements.txt
