# Dedup Module — Expansion for Entity Disambiguation

## Purpose of this document

The existing `nlp/dedup/` module performs article-level duplicate detection using MinHash
LSH + FAISS. This document covers the **additive expansion** for cross-document entity
disambiguation — mapping a local entity (produced by the Normalizer) to a canonical node
that already exists in the graph, or deciding that a new node should be created.

The existing article dedup code is **not changed**. Every existing function, class, and
endpoint keeps its current behaviour. New code is added alongside.

---

## What changes and what does not

| Component | Status |
|---|---|
| `minhash_index.py` | Unchanged |
| `embedding_index.py` | Unchanged |
| `id_map.py` | Unchanged |
| `persistence.py` | Unchanged |
| `service.py` — existing functions | Unchanged (`check`, `check_minhash_only`, `check_embedding_vec`, `bootstrap`) |
| `service.py` — new function | `entity_disambiguate()` added at bottom of file |
| `entity_index.py` | **New** — Weaviate-backed vector index for entities |
| `disambiguator.py` | **New** — three-tier decision logic + optional LLM adjudication |
| `api/routers/dedup.py` | **New endpoint** `/entity-disambiguate` appended; existing routes unchanged |
| `api/models.py` | **New models** for entity disambiguation appended; existing models unchanged |

---

## File layout (after expansion)

```
nlp/
  dedup/
    __init__.py
    minhash_index.py        ← unchanged
    embedding_index.py      ← unchanged
    id_map.py               ← unchanged
    persistence.py          ← unchanged
    service.py              ← unchanged functions + entity_disambiguate() appended
    entity_index.py         ← NEW: Weaviate-backed entity candidate index
    disambiguator.py        ← NEW: scoring + three-tier decision

api/
  routers/
    dedup.py                ← existing routes + /entity-disambiguate appended
```

---

## Cross-document entity disambiguation

### The problem

After the Normalizer produces `LocalEntity` objects (canonical within one document), we
need to answer: does "Acme Holdings Ltd" in document 42 refer to the same node as "Acme
Holdings" already in the graph from document 7? The answer determines whether to MERGE
into the existing node, CREATE a new one, or route to human REVIEW.

### Decision logic

```
Query Weaviate for top-K candidates filtered by entity type
         │
         ▼
  Score each candidate:
    - vector cosine similarity (name + description embedding)
    - alias exact match bonus (+0.15 if name in candidate.aliases)
    - type match (already filtered, but subtype mismatch = -0.05)
         │
         ▼
  best_score >= MERGE_THRESHOLD  →  "merge"   (high confidence)
  best_score in [REVIEW_LOW, MERGE_THRESHOLD) →  LLM adjudication
  best_score < REVIEW_LOW        →  "create"  (new canonical node)

  LLM adjudication result:
    "yes" → "merge"
    "no"  → "create"
    "unsure" → "review" (human queue)
```

Default thresholds (configurable via env):

| Threshold | Default | Meaning |
|---|---|---|
| `DISAMBIG_MERGE_THRESHOLD` | `0.92` | Above this → automatic merge |
| `DISAMBIG_REVIEW_LOW` | `0.75` | Below this → automatic create; above → LLM decides |
| `DISAMBIG_TOP_K` | `10` | Number of Weaviate candidates to retrieve |

---

## New files

### `nlp/dedup/entity_index.py`

Wraps the Weaviate client for entity candidate search. Does NOT manage the article FAISS
index (that stays in `embedding_index.py`).

```python
class EntityIndex:
    """Weaviate-backed nearest-neighbour index for canonical entity nodes."""

    def __init__(self, use_case: str, weaviate_client) -> None:
        self.use_case = use_case
        self._client = weaviate_client
        self._collection_name = f"Entity_{use_case}_v1"  # from ontology.weaviate_collection_spec()

    def find_candidates(
        self,
        embedding: list[float],
        entity_type: str,
        top_k: int = 10,
    ) -> list[Candidate]:
        """Return top-K candidates from Weaviate filtered by entity_type."""
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
                score=1.0 - obj.metadata.distance,   # convert distance → similarity
            )
            for obj in results.objects
        ]
```

`Candidate` is a dataclass:

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
```

The Weaviate client is injected (not module-level global) so tests can substitute a mock.
A module-level `_get_weaviate_client()` helper creates it from env vars and is called by
the service function.

### `nlp/dedup/disambiguator.py`

Pure orchestration logic. No module-level state (stateless, can run concurrently).

```python
@dataclass
class DisambiguationResult:
    decision:     Literal["merge", "create", "review"]
    canonical_id: str | None    # set when decision == "merge"
    confidence:   float
    candidates:   list[Candidate]

def disambiguate(
    local_entity:     LocalEntity,   # from Normalizer output
    embedding:        list[float],   # from entity encoder (bge-m3 via Ollama)
    entity_index:     EntityIndex,
    merge_threshold:  float,
    review_low:       float,
    top_k:            int,
    llm_adjudicate:   bool = True,
) -> DisambiguationResult:

    candidates = entity_index.find_candidates(embedding, local_entity.type, top_k)
    if not candidates:
        return DisambiguationResult("create", None, 1.0, [])

    best = _score(candidates, local_entity)

    if best.score >= merge_threshold:
        return DisambiguationResult("merge", best.canonical_id, best.score, candidates)

    if best.score < review_low or not llm_adjudicate:
        return DisambiguationResult(
            "create" if best.score < review_low else "review",
            None, best.score, candidates,
        )

    # Mid-confidence: ask the LLM
    verdict = _llm_adjudicate(local_entity, best)
    if verdict == "yes":
        return DisambiguationResult("merge", best.canonical_id, best.score, candidates)
    if verdict == "no":
        return DisambiguationResult("create", None, best.score, candidates)
    return DisambiguationResult("review", None, best.score, candidates)
```

**Scoring function** (`_score`): applies alias bonus (+0.15 if local entity name is in
`candidate.aliases`, capped at 1.0) and subtype penalty (-0.05 if subtypes differ).
Returns the highest-scoring candidate after adjustments.

**LLM adjudication** (`_llm_adjudicate`): single short prompt to Ollama asking whether
`local_entity.canonical_name` and `candidate.canonical_name` refer to the same real-world
entity, given both descriptions. Response schema: `{"verdict": "yes"|"no"|"unsure"}`.
Uses `EXTRACTION_MODEL` and a 30-second timeout. On Ollama error, returns `"unsure"` and
logs WARNING (non-fatal).

---

## Expansion to `service.py`

Append to the bottom of the existing `service.py` — the existing code above it is not
touched:

```python
# --- Entity disambiguation (new, independent of article dedup above) ---

import os as _os
from .entity_index import EntityIndex, _get_weaviate_client
from .disambiguator import disambiguate, DisambiguationResult

_MERGE_THRESHOLD  = float(_os.environ.get("DISAMBIG_MERGE_THRESHOLD", "0.92"))
_REVIEW_LOW       = float(_os.environ.get("DISAMBIG_REVIEW_LOW",      "0.75"))
_DISAMBIG_TOP_K   = int(  _os.environ.get("DISAMBIG_TOP_K",           "10"))

def entity_disambiguate(
    local_entity,
    embedding: list[float],
    use_case: str,
) -> dict:
    """Returns {decision, canonical_id, confidence, candidates[]}."""
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

---

## New API endpoint (`api/routers/dedup.py`)

Append to the existing router file after the existing routes:

```python
# --- Entity disambiguation ---

from api.models import EntityDisambiguateRequest, EntityDisambiguateResponse

@router.post("/entity-disambiguate", response_model=EntityDisambiguateResponse)
def entity_disambiguate(req: EntityDisambiguateRequest) -> EntityDisambiguateResponse:
    result = dedup_service.entity_disambiguate(
        local_entity=req.local_entity,
        embedding=req.embedding,
        use_case=req.use_case,
    )
    return EntityDisambiguateResponse(
        local_id=req.local_entity.local_id,
        **result,
    )
```

---

## New Pydantic models (`api/models.py`)

Append after existing models:

```python
# --- Entity Disambiguation ---

class EntityDisambiguateRequest(BaseModel):
    use_case:     str
    local_entity: LocalEntity       # from normalizer output
    embedding:    list[float]       # bge-m3 vector, 1024-dim

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

## Entity embedding

Entity disambiguation requires a different embedding model from the article embedding
(`paraphrase-multilingual-MiniLM-L12-v2`, 384-dim). The entity embedding uses `bge-m3`
(1024-dim) via Ollama's `/api/embed` endpoint, which the caller (the orchestrating pipeline
or the client submitting to `/entity-disambiguate`) is responsible for computing.

The `/entity-disambiguate` endpoint accepts a pre-computed `embedding: list[float]` so:
- The caller decides which model produced it (as long as it matches what Weaviate stores)
- The NLP service does not load a second embedding model into process memory

A thin helper `nlp/entity_encoder.py` is provided for internal use and testing:

```python
# nlp/entity_encoder.py
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

---

## Weaviate setup

The Weaviate collection for each use case is created by the persistence module (out of
scope for this document) using `ontology.weaviate_collection_spec()`. The disambiguator
assumes the collection already exists and reads from it only.

---

## Configuration

| Env var | Default | Description |
|---|---|---|
| `DISAMBIG_MERGE_THRESHOLD` | `0.92` | Auto-merge above this similarity |
| `DISAMBIG_REVIEW_LOW` | `0.75` | Auto-create below this; LLM adjudicates in between |
| `DISAMBIG_TOP_K` | `10` | Weaviate candidate count |
| `WEAVIATE_HOST` | `http://weaviate:8080` | Weaviate HTTP endpoint |
| `WEAVIATE_GRPC_PORT` | `50051` | Weaviate gRPC port |
| `ENTITY_EMBED_MODEL` | `bge-m3` | Ollama model for entity embeddings |

---

## Dependencies

- `weaviate-client>=4.0` — **new package** (Weaviate Python v4 SDK)
- `nlp/ontology.py`
- `nlp/entity_encoder.py`
- Ollama sidecar (LLM adjudication + entity embedding)
- `httpx` — already in requirements.txt
