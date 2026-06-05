# Dedup — Implementation Doc

**Module:** `nlp/dedup/` · **Status:** Reworked (FAISS/local-state path retired → Weaviate)
**Authoritative source:** `docs/superpowers/specs/2026-06-05-nlp-two-pipelines-design.md` §3.6

## Purpose

Cross-document dedup against **Weaviate**, in **two distinct modes**:

- **`/dedup/item`** — one new item vs an existing collection → **match / no-match** (a 2-step
  decision: similarity threshold, then LLM double-check in the mid band). Used by the news
  pipeline at the embedding stage (doc 07, step 5).
- **`/dedup/corpus`** — a whole collection (restricted to entities or relations) → **clusters**.
  Used by the relation pipeline (doc 08) for entity and relation dedup.

The module is **read-only on Weaviate** and **decides, it does not write**: the orchestrator owns
collection creation, inserts, and merges. Embeddings are **bge-m3 (1024-dim)** computed by the
`vectorizer` (spec §3.7); `/dedup/item` receives a precomputed vector, `/dedup/corpus` reads the
vectors already stored in the collection.

> **MinHash moves to the orchestrator.** The cheap article MinHash check (news pipeline step 1)
> is the orchestrator's job now (doc 07), querying minhash signatures stored as a Weaviate
> property. This module keeps only the stateless MinHash **primitive** (`minhash.py`) the
> orchestrator calls; the in-process LSH index, FAISS, the id map, and local pickle persistence
> are all retired.

---

## What changes and what does not

| Component | Status |
|---|---|
| `nlp/dedup/minhash_index.py` | **Reduced → `minhash.py`**: keep `compute()`/shingling, add `jaccard()`; **delete the `MinHashIndex` LSH class** |
| `nlp/dedup/embedding_index.py` (FAISS) | **Deleted** (→ Weaviate) |
| `nlp/dedup/id_map.py` | **Deleted** (FAISS row mapping no longer needed) |
| `nlp/dedup/persistence.py` | **Deleted** (no local pickled state) |
| `nlp/dedup/service.py` — `check`, `check_minhash_only`, `check_embedding_vec`, `bootstrap`, `load`, `flush` | **Deleted** (minhash → orchestrator; embedding → `/dedup/item`) |
| `nlp/dedup/weaviate_index.py` | **New** — Weaviate near-vector query wrapper |
| `nlp/dedup/item.py` | **New** — single-item 2-step decision |
| `nlp/dedup/corpus.py` | **New** — collection clustering |
| `nlp/dedup/llm.py` | **New** — LLM adjudication (shared by item + corpus) |
| `nlp/dedup/service.py` | **Rewritten** — `dedup_item()`, `dedup_corpus()` (thin orchestration) |
| `api/routers/dedup.py` — `/dedup-check`, `/dedup-check-embed`, `/dedup/bootstrap` | **Deleted** |
| `api/routers/dedup.py` — `/dedup/item`, `/dedup/corpus` | **New** |
| `api/models.py` — `DedupRequest`, `DedupCheckEmbedRequest`, `DedupResponse`, `Bootstrap*` | **Deleted** |
| `api/models.py` — `DedupItem*`, `DedupCorpus*`, `Candidate` | **New** |
| `api/main.py` warmup | Drop dedup `load()`/FAISS warmup + SIGTERM `flush()`; nothing to preload |
| `requirements.txt` | **Add** `weaviate-client>=4`; **remove** `faiss-cpu` if unused elsewhere |

> The earlier entity-disambiguation sketch (a Weaviate candidate index + tiered thresholds + LLM
> adjudication) is **generalised** here into `/dedup/item` (any collection) and `/dedup/corpus`,
> and the article path is folded in rather than kept on FAISS.

> Blast radius before deleting the old endpoints:
> `grep -rn "dedup-check\|dedup/bootstrap\|check_minhash_only\|check_embedding_vec" . ` — expect
> hits only in `api/routers/dedup.py`, `ingestion/07_news/`, and `eval/05_dedup_eval.ipynb`. The
> ingestion driver becomes the news orchestrator (doc 07); the `eval/` notebooks are **deprecated**
> (test-only) and will be replaced by a new test suite designed later.

---

## File layout (after rework)

```
nlp/dedup/
  __init__.py
  minhash.py          ← compute(text)->list[int], jaccard(a,b)->float   (stateless util)
  weaviate_index.py   ← WeaviateIndex: near_vector queries + fetch_all (read-only)
  item.py             ← decide_item(): 2-step single-item decision
  corpus.py           ← cluster(): collection clustering
  llm.py              ← adjudicate(): yes/no/unsure LLM call
  service.py          ← dedup_item(), dedup_corpus()
api/routers/dedup.py  ← /dedup/item, /dedup/corpus
```

---

## MinHash primitive (`nlp/dedup/minhash.py`)

Keep the existing shingling + `datasketch` MinHash computation; expose two **stateless**
functions. Drop the `MinHashIndex` LSH class — there is no in-process index anymore.

```python
import os, unicodedata
from datasketch import MinHash

_NUM_PERM = int(os.environ.get("MINHASH_NUM_PERM", "128"))
_SHINGLE_SIZE = 3

def _normalize(text: str) -> str:
    text = text.lower()
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")

def _shingles(text: str) -> list[str]:
    words = _normalize(text).split()
    if len(words) < _SHINGLE_SIZE:
        return [" ".join(words)] if words else []
    return [" ".join(words[i:i + _SHINGLE_SIZE]) for i in range(len(words) - _SHINGLE_SIZE + 1)]

def compute(text: str) -> list[int]:
    """Return the MinHash signature as a plain int list — storable as a Weaviate property."""
    mh = MinHash(num_perm=_NUM_PERM)
    for s in _shingles(text):
        mh.update(s.encode("utf-8"))
    return mh.hashvalues.tolist()

def jaccard(a: list[int], b: list[int]) -> float:
    """Estimated Jaccard from two equal-length signatures."""
    if not a or len(a) != len(b):
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)
```

The orchestrator (doc 07) stores `compute(text)` on each article in Weaviate and, for a new
article, fetches candidate signatures and uses `jaccard()` to find easy duplicates before any LLM
or embedding work.

---

## Weaviate query wrapper (`nlp/dedup/weaviate_index.py`)

```python
import os
import weaviate
from weaviate.classes.query import Filter, MetadataQuery
from dataclasses import dataclass

@dataclass
class Candidate:
    id:    str
    score: float                 # cosine similarity (1 - distance)
    props: dict                  # returned properties (e.g. canonical_name, description, summary)


class WeaviateIndex:
    """Read-only nearest-neighbour access to one Weaviate collection."""

    def __init__(self, collection: str, client=None) -> None:
        self.collection = collection
        self._client = client or _get_client()

    def near(self, embedding, top_k, filters=None, return_props=()) -> list[Candidate]:
        res = self._client.collections.get(self.collection).query.near_vector(
            near_vector=embedding, limit=top_k, filters=filters,
            return_properties=list(return_props), return_metadata=MetadataQuery(distance=True),
            include_vector=False,
        )
        return [Candidate(id=str(o.uuid), score=1.0 - o.metadata.distance, props=o.properties)
                for o in res.objects]

    def fetch_all(self, filters=None, return_props=(), with_vector=True):
        """Iterate every object (optionally filtered) — for corpus clustering."""
        coll = self._client.collections.get(self.collection)
        for o in coll.iterator(include_vector=with_vector,
                               return_properties=list(return_props)):
            yield o   # o.uuid, o.properties, o.vector


def _get_client():
    return weaviate.connect_to_custom(
        http_host=os.environ.get("WEAVIATE_HTTP_HOST", "weaviate"),
        http_port=int(os.environ.get("WEAVIATE_HTTP_PORT", "8080")),
        http_secure=False,
        grpc_host=os.environ.get("WEAVIATE_GRPC_HOST", "weaviate"),
        grpc_port=int(os.environ.get("WEAVIATE_GRPC_PORT", "50051")),
        grpc_secure=False,
    )
```

The client is injectable so tests pass a fake. A type/scope filter is built with
`Filter.by_property("type").equal(...)` by the caller.

---

## Mode A — `/dedup/item` (single item, 2-step)

```python
# nlp/dedup/item.py
from dataclasses import dataclass
from typing import Literal

@dataclass
class ItemDecision:
    decision:   Literal["match", "no_match"]
    target_id:  str | None       # set when decision == "match"
    score:      float
    candidates: list             # Candidate list for transparency


def decide_item(index, embedding, *, kind, compare_text, compare_property,
                match_threshold, llm_low, top_k, filters=None, llm=True) -> ItemDecision:
    cands = index.near(embedding, top_k, filters=filters,
                       return_props=(compare_property,))
    if not cands:
        return ItemDecision("no_match", None, 0.0, [])
    best = cands[0]

    if best.score >= match_threshold:                       # step 1: high similarity → match
        return ItemDecision("match", best.id, best.score, cands)
    if best.score < llm_low or not llm:                     # below band → no match
        return ItemDecision("no_match", None, best.score, cands)

    # step 2: mid band → LLM double-check
    verdict = adjudicate(kind, compare_text, best.props.get(compare_property, ""))
    decision = "match" if verdict == "yes" else "no_match"  # unsure → no_match (conservative)
    return ItemDecision(decision, best.id if decision == "match" else None, best.score, cands)
```

For the news pipeline: `kind="article"`, `compare_property="summary"`, `compare_text` is the new
article's `headline + summary`, `filters=None`. The decision is binary (no "review" queue — the
news pipeline either merges into the matched article or continues). The orchestrator performs the
merge (append source+url, oldest date) on `match`.

## Mode B — `/dedup/corpus` (whole collection → clusters)

```python
# nlp/dedup/corpus.py
def cluster(index, *, kind, compare_property, merge_threshold, llm_low, top_k, filters=None,
            llm=True) -> list[list[str]]:
    """Union-find over near-neighbour edges. Returns clusters of object ids (singletons included)."""
    objs = list(index.fetch_all(filters=filters, return_props=(compare_property,), with_vector=True))
    uf = _UnionFind(o.uuid for o in objs)

    for o in objs:
        for cand in index.near(o.vector, top_k, filters=filters, return_props=(compare_property,)):
            if cand.id == str(o.uuid):
                continue
            if cand.score >= merge_threshold:                       # auto-merge
                uf.union(str(o.uuid), cand.id)
            elif cand.score >= llm_low and llm:                     # mid band → LLM
                if adjudicate(kind, o.properties.get(compare_property, ""),
                              cand.props.get(compare_property, "")) == "yes":
                    uf.union(str(o.uuid), cand.id)
    return uf.groups()
```

Run once with `filters` selecting **entities**, once selecting **relations** (doc 08). Returns
clusters (lists of Weaviate ids); the orchestrator merges each cluster and re-describes it with
`summarizer[aggregate]`. `_UnionFind` is a tiny pure-Python helper (`find`/`union`/`groups`).

## LLM adjudication (`nlp/dedup/llm.py`)

```python
def adjudicate(kind: str, text_a: str, text_b: str) -> str:
    """Ask the LLM whether two items are the same. Returns 'yes' | 'no' | 'unsure'.

    `kind` flavours the prompt ('article' | 'entity' | 'relation'); the logic is identical.
    On Ollama error returns 'unsure' (non-fatal) and logs WARNING.
    """
    ...  # single short /api/chat call, schema {"verdict": "yes"|"no"|"unsure"}, EXTRACTION_MODEL
```

Generic by design: it compares two texts. The caller supplies the comparable text (article
summary, entity name+description, relation evidence) so the module stays agnostic to schema.

---

## Service + API

```python
# nlp/dedup/service.py
def dedup_item(collection, embedding, *, kind, compare_text, compare_property="summary",
               type_filter=None) -> dict:
    idx = WeaviateIndex(collection)
    filters = Filter.by_property("type").equal(type_filter) if type_filter else None
    d = decide_item(idx, embedding, kind=kind, compare_text=compare_text,
                    compare_property=compare_property,
                    match_threshold=_ITEM_MATCH, llm_low=_ITEM_LLM_LOW,
                    top_k=_TOP_K, filters=filters)
    return {"decision": d.decision, "target_id": d.target_id, "score": d.score,
            "candidates": [vars(c) for c in d.candidates]}

def dedup_corpus(collection, *, kind, compare_property, type_filter) -> dict:
    idx = WeaviateIndex(collection)
    filters = Filter.by_property("type").equal(type_filter) if type_filter else None
    clusters = cluster(idx, kind=kind, compare_property=compare_property,
                       merge_threshold=_CORPUS_MERGE, llm_low=_CORPUS_LLM_LOW,
                       top_k=_TOP_K, filters=filters)
    return {"clusters": clusters}
```

```python
# api/routers/dedup.py
@router.post("/dedup/item", response_model=DedupItemResponse)
def dedup_item(req: DedupItemRequest) -> DedupItemResponse:
    return DedupItemResponse(request_id=req.request_id, **dedup_service.dedup_item(
        req.collection, req.embedding, kind=req.kind, compare_text=req.compare_text,
        compare_property=req.compare_property, type_filter=req.type_filter))

@router.post("/dedup/corpus", response_model=DedupCorpusResponse)
def dedup_corpus(req: DedupCorpusRequest) -> DedupCorpusResponse:
    return DedupCorpusResponse(request_id=req.request_id, **dedup_service.dedup_corpus(
        req.collection, kind=req.kind, compare_property=req.compare_property,
        type_filter=req.type_filter))
```

### Models (`api/models.py`)

```python
class DedupItemRequest(BaseModel):
    request_id:       str | None = None
    collection:       str
    embedding:        list[float]
    kind:             str = "article"          # article | entity | relation (LLM prompt flavour)
    compare_text:     str                      # new item's comparable text (for LLM step)
    compare_property: str = "summary"          # stored property read for candidates
    type_filter:      str | None = None        # restrict candidates by `type`

class Candidate(BaseModel):
    id: str; score: float; props: dict

class DedupItemResponse(BaseModel):
    request_id: str | None = None
    decision:   Literal["match", "no_match"]
    target_id:  str | None
    score:      float
    candidates: list[Candidate]

class DedupCorpusRequest(BaseModel):
    request_id:       str | None = None
    collection:       str
    kind:             str = "entity"
    compare_property: str = "description"
    type_filter:      str | None = None        # "entity" vs "relation" subset

class DedupCorpusResponse(BaseModel):
    request_id: str | None = None
    clusters:   list[list[str]]                # each inner list = one cluster of object ids
```

---

## Configuration

| Env var | Default | Description |
|---|---|---|
| `DEDUP_ITEM_MATCH` | `0.92` | Auto-match above this cosine similarity |
| `DEDUP_ITEM_LLM_LOW` | `0.75` | Below → no-match; between → LLM double-check |
| `DEDUP_CORPUS_MERGE` | `0.92` | Auto-merge edge in corpus clustering |
| `DEDUP_CORPUS_LLM_LOW` | `0.80` | Mid-band edge → LLM check |
| `DEDUP_TOP_K` | `10` | Neighbours fetched per query |
| `MINHASH_NUM_PERM` | `128` | MinHash permutations (orchestrator's signature length) |
| `WEAVIATE_HTTP_HOST` / `WEAVIATE_HTTP_PORT` | `weaviate` / `8080` | Weaviate HTTP |
| `WEAVIATE_GRPC_HOST` / `WEAVIATE_GRPC_PORT` | `weaviate` / `50051` | Weaviate gRPC |
| `EXTRACTION_MODEL` | `qwen2.5:32b` | LLM for adjudication |

Removed: `DEDUP_PERSIST_EVERY_N`, `DEDUP_LSH_THRESHOLD`, FAISS/embedding env (local index gone).

---

## Dependencies

- `weaviate-client>=4` — **new package** (item queries + corpus iteration).
- `datasketch` — already in requirements (MinHash primitive).
- Ollama sidecar — LLM adjudication.
- Remove `faiss-cpu` from requirements if nothing else uses it.
