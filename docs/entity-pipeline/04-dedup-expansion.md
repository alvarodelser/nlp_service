# Dedup Module — Weaviate-Backed Rewrite

## Purpose of this document

The existing `nlp/dedup/` module performs article-level duplicate detection using
in-process MinHash LSH and FAISS indexes. This document covers the **full replacement**
of that implementation with a stateless, Weaviate-backed design, and the addition of
cross-document entity disambiguation as a second dedup path in the same module.

Both paths work identically: the service receives a document or entity plus a
pre-computed embedding, queries Weaviate, scores the candidates, and returns a decision.
No in-process indexes. No disk state. No global variables.

---

## What changes

| Component | Old | New |
|---|---|---|
| `minhash_index.py` | MinHash LSH in-process index | **Deleted** |
| `embedding_index.py` | FAISS in-process index | **Deleted** |
| `id_map.py` | row-to-article_id map | **Deleted** |
| `persistence.py` | pickle flush/load | **Deleted** |
| `service.py` | stateful, global `_mh/_emb/_ids`, thread lock | **Rewritten** — stateless |
| `article_index.py` | did not exist | **New** — Weaviate-backed article dedup |
| `entity_index.py` | did not exist | **New** — Weaviate-backed entity candidate search |
| `disambiguator.py` | did not exist | **New** — three-tier entity decision logic |
| `api/routers/dedup.py` | `POST /dedup` | `POST /dedup` updated + `POST /entity-disambiguate` added |
| `api/models.py` | existing dedup models | existing unchanged + new entity disambiguation models |

---

## File layout (after rewrite)

```
nlp/
  dedup/
    __init__.py
    article_index.py    ← NEW: Weaviate-backed article near-duplicate search
    entity_index.py     ← NEW: Weaviate-backed entity candidate search
    disambiguator.py    ← NEW: scoring + three-tier entity decision
    service.py          ← REWRITTEN: stateless functions, no global state

api/
  routers/
    dedup.py            ← POST /dedup (updated) + POST /entity-disambiguate (added)
```

---

## Article dedup

### How it works

The caller (orchestrator or ingestion pipeline) pre-computes an embedding for the article
text using `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, same model as before) and
sends it to `POST /dedup`. The NLP service queries the Weaviate `Articles` collection for
vectors above the similarity threshold. If a match is found the article is a duplicate and
is not indexed. If no match is found the article is added to Weaviate and marked as
canonical.

The query and the insert happen in the same service call so the caller does not need to
make a separate write request.

### `nlp/dedup/article_index.py`

```python
class ArticleIndex:
    """Weaviate-backed near-duplicate index for articles."""

    COLLECTION = "Articles"

    def __init__(self, weaviate_client) -> None:
        self._client = weaviate_client

    def find_near_duplicate(
        self,
        embedding: list[float],
        threshold: float,
    ) -> tuple[str | None, float]:
        """Return (article_id, score) of the nearest existing article, or (None, 0.0)."""
        results = (
            self._client.collections.get(self.COLLECTION)
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
        if score < threshold:
            return None, 0.0
        return obj.properties["article_id"], score

    def add(self, article_id: str, embedding: list[float]) -> None:
        """Insert a new canonical article embedding."""
        self._client.collections.get(self.COLLECTION).data.insert(
            properties={"article_id": article_id},
            vector=embedding,
        )
```

The `Articles` Weaviate collection must exist before the service starts. It requires a
single property `article_id: text` and a cosine vector index (384-dim).

### `service.py` — article dedup function

```python
def check(article_id: str, embedding: list[float]) -> dict:
    """Query Weaviate for near-duplicates; insert if none found.

    Returns {duplicate_of, score, indexed}.
    """
    idx = ArticleIndex(_get_weaviate_client())
    dup_id, score = idx.find_near_duplicate(embedding, _ARTICLE_THRESHOLD)
    if dup_id is not None:
        return {"duplicate_of": dup_id, "score": score, "indexed": False}
    idx.add(article_id, embedding)
    return {"duplicate_of": None, "score": None, "indexed": True}
```

### API — `POST /dedup`

The existing request model gains an `embedding` field. The `text` field is retained for
backward compatibility but is no longer used for Stage 2 (embedding check) — the caller
provides the vector. If `embedding` is absent the endpoint falls back to embedding the
`text` inline (using the same MiniLM model).

```python
class DedupRequest(BaseModel):
    article_id: str
    text:       str
    embedding:  list[float] | None = None   # pre-computed; if absent, computed inline

class DedupResponse(BaseModel):
    duplicate_of: str | None
    score:        float | None
    indexed:      bool
```

---

## Entity disambiguation

### How it works

After the Clusterer produces `LocalEntity` objects (canonical within one document), the
orchestrator computes a `bge-m3` embedding (1024-dim) for each entity (name + description)
and sends it to `POST /entity-disambiguate`. The service queries the Weaviate
`Entity_{use_case}_v1` collection, scores the candidates, and returns a three-tier decision.

The service **does not write to Weaviate** on this path — the orchestrator decides when to
create or merge nodes.

### Decision logic

```
Query Weaviate for top-K candidates filtered by entity type
         │
         ▼
  Score each candidate:
    - base score: cosine similarity from Weaviate (1.0 - distance)
    - alias bonus: +0.15 if local entity name is in candidate.aliases (capped at 1.0)
    - subtype penalty: -0.05 if subtypes differ
         │
         ▼
  best_score >= DISAMBIG_MERGE_THRESHOLD   →  "merge"   (automatic)
  best_score in [DISAMBIG_REVIEW_LOW, DISAMBIG_MERGE_THRESHOLD)  →  LLM adjudication
  best_score < DISAMBIG_REVIEW_LOW         →  "create"  (automatic)

  LLM adjudication result:
    "yes"    → "merge"
    "no"     → "create"
    "unsure" → "review"
```

| Threshold env var | Default | Meaning |
|---|---|---|
| `DISAMBIG_MERGE_THRESHOLD` | `0.92` | Above → automatic merge |
| `DISAMBIG_REVIEW_LOW` | `0.75` | Below → automatic create |
| `DISAMBIG_TOP_K` | `10` | Weaviate candidate count |

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
    score:          float        # cosine similarity from Weaviate

class EntityIndex:
    """Weaviate-backed nearest-neighbour index for canonical entity nodes."""

    def __init__(self, use_case: str, weaviate_client) -> None:
        self.use_case = use_case
        self._client = weaviate_client
        self._collection_name = f"Entity_{use_case}_v1"

    def find_candidates(
        self,
        embedding: list[float],
        entity_type: str,
        top_k: int = 10,
    ) -> list[Candidate]:
        results = (
            self._client.collections.get(self._collection_name)
            .query.near_vector(
                near_vector=embedding,
                limit=top_k,
                filters=Filter.by_property("type").equal(entity_type),
                return_properties=["canonical_name", "type", "subtype", "description", "aliases"],
                return_metadata=MetadataQuery(distance=True),
            )
        )
        return [
            Candidate(
                canonical_id=str(obj.uuid),
                canonical_name=obj.properties["canonical_name"],
                type=obj.properties["type"],
                subtype=obj.properties.get("subtype"),
                description=obj.properties.get("description", ""),
                aliases=obj.properties.get("aliases", []),
                score=1.0 - obj.metadata.distance,
            )
            for obj in results.objects
        ]
```

### `nlp/dedup/disambiguator.py`

Pure logic — no Weaviate or LLM dependency (those are injected). Fully testable in
isolation.

```python
@dataclass
class DisambiguationResult:
    decision:     Literal["merge", "create", "review"]
    canonical_id: str | None    # set when decision == "merge"
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

    best = _score(candidates, local_entity)

    if best.score >= merge_threshold:
        return DisambiguationResult("merge", best.canonical_id, best.score, candidates)

    if best.score < review_low or not llm_adjudicate:
        decision = "create" if best.score < review_low else "review"
        return DisambiguationResult(decision, None, best.score, candidates)

    verdict = _llm_adjudicate(local_entity, best)
    if verdict == "yes":
        return DisambiguationResult("merge", best.canonical_id, best.score, candidates)
    if verdict == "no":
        return DisambiguationResult("create", None, best.score, candidates)
    return DisambiguationResult("review", None, best.score, candidates)
```

`_score` applies alias bonus (+0.15, capped at 1.0) and subtype penalty (−0.05). Returns
the highest-scoring candidate after adjustments.

`_llm_adjudicate` sends a single short prompt to Ollama asking whether two entity names
and descriptions refer to the same real-world entity. Response schema:
`{"verdict": "yes"|"no"|"unsure"}`. Uses `EXTRACTION_MODEL` and a 30-second timeout.
On Ollama error returns `"unsure"` and logs WARNING (non-fatal — the decision becomes
`"review"`).

### `service.py` — entity disambiguation function

```python
def entity_disambiguate(
    local_entity: LocalEntity,
    embedding:    list[float],
    use_case:     str,
) -> dict:
    idx = EntityIndex(use_case, _get_weaviate_client())
    result = disambiguate(
        local_entity=local_entity,
        embedding=embedding,
        entity_index=idx,
        merge_threshold=_MERGE_THRESHOLD,
        review_low=_REVIEW_LOW,
        top_k=_DISAMBIG_TOP_K,
    )
    return {
        "decision":     result.decision,
        "canonical_id": result.canonical_id,
        "confidence":   result.confidence,
        "candidates":   [vars(c) for c in result.candidates],
    }
```

### API — `POST /entity-disambiguate`

```python
class EntityDisambiguateRequest(BaseModel):
    use_case:     str
    local_entity: LocalEntity
    embedding:    list[float]   # bge-m3 vector, 1024-dim

class CandidateResult(BaseModel):
    canonical_id:   str
    canonical_name: str
    type:           str
    subtype:        str | None
    description:    str
    aliases:        list[str]
    score:          float

class EntityDisambiguateResponse(BaseModel):
    local_id:     str
    decision:     Literal["merge", "create", "review"]
    canonical_id: str | None
    confidence:   float
    candidates:   list[CandidateResult]
```

---

## Weaviate client helper

Both paths share the same client factory function in `service.py`:

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

The client is created per-request (connection pool is managed by the Weaviate SDK
internally). No module-level client singleton.

---

## Entity embedding

Entity disambiguation requires `bge-m3` (1024-dim). The caller is responsible for
computing the embedding via Ollama `/api/embed` before calling `/entity-disambiguate`.

A thin helper `nlp/entity_encoder.py` is available for internal use and testing:

```python
def embed(texts: list[str]) -> list[list[float]]:
    """Embed via Ollama /api/embed using bge-m3. Returns list of 1024-dim vectors."""
    response = httpx.post(
        f"{OLLAMA_HOST}/api/embed",
        json={"model": ENTITY_EMBED_MODEL, "input": texts},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()["embeddings"]
```

| Env var | Default |
|---|---|
| `ENTITY_EMBED_MODEL` | `bge-m3` |

Article embeddings continue to use `paraphrase-multilingual-MiniLM-L12-v2` (384-dim),
computed by the caller or inline in the `/dedup` endpoint if no `embedding` is provided.

---

## Configuration

| Env var | Default | Description |
|---|---|---|
| `DEDUP_ARTICLE_THRESHOLD` | `0.85` | Cosine similarity threshold for article duplicates |
| `DISAMBIG_MERGE_THRESHOLD` | `0.92` | Entity: auto-merge above this |
| `DISAMBIG_REVIEW_LOW` | `0.75` | Entity: auto-create below this |
| `DISAMBIG_TOP_K` | `10` | Weaviate candidate count for entity search |
| `WEAVIATE_HOST` | `weaviate` | Weaviate hostname |
| `WEAVIATE_PORT` | `8080` | Weaviate HTTP port |
| `WEAVIATE_GRPC_PORT` | `50051` | Weaviate gRPC port |
| `ENTITY_EMBED_MODEL` | `bge-m3` | Ollama model for entity embeddings |

---

## Testing

See [06-testing.md](06-testing.md) §3 — `disambiguator.py` is pure logic with no
Weaviate or Ollama calls; candidates are passed in directly. `article_index.py` and
`entity_index.py` are tested with an injected mock Weaviate client. Key cases: automatic
merge, automatic create, LLM adjudication (each verdict), alias bonus, subtype penalty,
empty candidate list, Weaviate unavailable.

---

## Dependencies

- `weaviate-client>=4.0` — **new package** (replaces `faiss-cpu` which is removed)
- `httpx` — already in requirements.txt
- `sentence-transformers` — kept for inline article embedding fallback
- `faiss-cpu` — **removed**
