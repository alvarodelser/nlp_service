# Resolver — Implementation Doc

**Module:** `nlp/resolver/` · **Status:** New (written from scratch)
**Authoritative source:** `docs/superpowers/specs/2026-06-05-nlp-two-pipelines-design.md` §3.5

## Purpose

A **stateless, intra-document resolution service**: given a document's entity mentions **and the
relations between them**, reduce the mentions to canonical entities (each with an assigned id) and
rewrite the relations onto those canonical entities.

It is intra-document by nature — the orchestrator calls it once per document — so there is **no
`doc_id`** and nothing about chunks, offsets, use-case, or schema. Entities and relations come in
referencing each other by **name** (the orchestrator flattened the per-chunk `ner` output and
turned each relation's endpoint indices into the endpoint entity's name).

```
  IN   entities:  [{name, type, subtype?, evidence}, ...]
       relations: [{head:name, tail:name, type, subtype?, evidence, attributes, confidence}, ...]
                       │
                       ▼
              ┌────────────────────┐
              │      resolver       │  pre-merge → LLM refine → assign ids → rewrite relations
              └────────────────────┘
                       │
                       ▼
  OUT  entities:  [{id, canonical_name, names[], type, subtype, evidence[]}, ...]   ← reduced
       relations: [{head_id, tail_id, type, subtype, evidence[], attributes, confidence}, ...]
```

> Coreference (classical "entity resolution") runs here, **after** extraction, on the structured
> output — not as a pre-processing step on raw text. Pronoun→name resolution happened upstream in
> `ner.coref` before chunking (doc 04); cross-document disambiguation happens later in
> `dedup/corpus` (doc 06). These are three separate scopes.

---

## File layout

```
nlp/
  resolver/
    __init__.py
    service.py        ← pre-merge → refine → assign ids → rewrite relations → return
    premerge.py       ← pure-Python exact-name (optionally embedding) blocking
    edges.py          ← pure-Python relation rewrite + overlap collapse (no LLM)
    ollama_client.py  ← /api/chat with the cluster schema
    prompts/
      resolution.txt
api/
  routers/
    resolve.py        ← POST /resolve
```

---

## Data contracts

### Input — `ResolveRequest`

```python
class EntityIn(BaseModel):
    name:     str            # surface name as extracted ("Laura Méndez", "Méndez", "la ministra")
    type:     str
    subtype:  str | None = None
    evidence: str            # the sentence the mention appeared in

class RelationIn(BaseModel):
    head:       str          # subject entity name
    tail:       str          # object entity name
    type:       str
    subtype:    str | None = None
    evidence:   str          # the clause stating the relation
    attributes: dict
    confidence: float

class ResolveRequest(BaseModel):
    entities:  list[EntityIn]
    relations: list[RelationIn]
```

No `doc_id`, no ids, no offsets. Relations point at entities by name; the service maps each name
to its canonical entity (and its assigned id).

### Output — `ResolveResponse`

```python
class ResolvedEntity(BaseModel):
    id:             str             # assigned local id (stable within this response)
    canonical_name: str             # chosen name for the reduced entity
    names:          list[str]       # every surface name merged into it (its aliases)
    type:           str
    subtype:        str | None
    evidence:       list[str]       # every member's evidence sentence, collected

class ResolvedRelation(BaseModel):
    head_id:    str                 # a ResolvedEntity.id
    tail_id:    str                 # a ResolvedEntity.id
    type:       str
    subtype:    str | None
    evidence:   list[str]           # collected; >1 only when overlap duplicates collapsed
    attributes: dict                # passed through, never aggregated
    confidence: float               # max across collapsed duplicates

class ResolveResponse(BaseModel):
    entities:  list[ResolvedEntity]
    relations: list[ResolvedRelation]
```

The reduced entity keeps its `canonical_name`, the `names` it absorbed (so the caller can match
anything back), and an **assigned `id`**. Relations reference entities **by id**, which sidesteps
the rare same-canonical-name collision (see Edge cases) — the orchestrator carries these ids
straight into `summarizer[entity_desc/relation_desc]` (doc 01) and the per-document Weaviate
upsert.

---

## Algorithm

### Step 1 — Pre-merge entities (pure Python, no LLM) — `premerge.py`

Normalize each entity's name (lowercase; strip honorifics, punctuation, legal suffixes like
"S.A."/"Ltd") and group entities that share a normalized name **and** type into one **candidate**.
Most entries are literal repeats ("Laura Méndez"/"Méndez" 20×), so this collapses a long list into
a handful of candidates, each keeping its member names, a provisional canonical name (most
complete form), and one representative evidence sentence.

> Optionally block by `bge-m3` embedding similarity instead of exact form, to catch spelling
> variants ("Banco Santander"/"Santander S.A."). Reuses the `vectorizer` (spec §3.7).

### Step 2 — LLM refine (the only LLM call)

Send the **candidates** (not the raw entities) to the LLM, each with its provisional name, type,
and representative evidence. It does only what pre-merge cannot: merge a definite description or
pronoun-leftover mention into a named candidate, and split same-named different entities when
their evidence diverges. It references candidates by integer index.

```jsonc
{
  "type": "object",
  "required": ["clusters"],
  "properties": {
    "clusters": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["canonical_name", "type", "subtype", "candidate_indices"],
        "additionalProperties": false,
        "properties": {
          "canonical_name":    {"type": "string"},
          "type":              {"type": "string"},
          "subtype":           {"oneOf": [{"type": "string"}, {"type": "null"}]},
          "candidate_indices": {"type": "array", "items": {"type": "integer"}}
        }
      }
    }
  }
}
```

**Skip the LLM entirely** when pre-merge yields all unique-named singletons. If candidates exceed
`RESOLVE_CANDIDATE_BATCH`, refine per type then merge across types only where names match. The
LLM input scales with *candidate* count, not entity count — so it stays small even for very long
documents.

### Step 3 — Assign ids + build the name map (pure Python)

For each refined cluster, emit a `ResolvedEntity` with an assigned `id` (sequential within this
response: `"0"`, `"1"`, …), its `canonical_name`, all member `names`, type/subtype, and collected
`evidence`. Build `name → id` from the members — the lookup Step 4 uses.

### Step 4 — Rewrite + de-duplicate relations (pure Python, no LLM) — `edges.py`

1. **Rewrite endpoints:** map each relation's `head`/`tail` name to the resolved entity's `id` via
   the Step 3 map. A name in no cluster → log WARNING, drop the relation.
2. **Collapse overlap duplicates:** group by `(head_id, type, tail_id, evidence_fingerprint)`;
   relations with the same endpoints **and** overlapping evidence (≥ 80 %, the same sentence from
   two overlapping chunks) collapse into one — keeping both evidence snippets, `confidence` = max,
   attributes from the highest-confidence instance.
3. **Distinct evidence = distinct relations.** Two payments between the same parties are two
   payments. Attributes are **never** aggregated (`[[project_normalizer_edge_semantics]]`).

---

## Prompt template (`prompts/resolution.txt`)

```
You are resolving entities to canonical identities.
You will receive a numbered list of CANDIDATE entities (already pre-merged from exact-name
matches), each with a provisional name, type, and one example sentence (evidence). Merge the
candidates that refer to the same real-world entity into clusters.

RULES
=====
- Every candidate goes into exactly one cluster (a candidate alone forms a singleton).
- Merge a candidate into another only when the evidence makes it the SAME entity — e.g. a
  definite description ("the minister") or pronoun mention matching a named candidate.
- Split candidates that share a name but are clearly different entities by their evidence.
- Choose the most complete, unambiguous name as canonical_name. Preserve original language/spelling.
- Only cluster candidates of the same type.
- Reference candidates by their integer index (0-based) in candidate_indices.
- Do NOT write a description — that is produced later by the summarizer.

CANDIDATES
==========
{candidate_list}
```

---

## Edge cases

| Situation | Handling |
|---|---|
| Pre-merge yields all unique-named singletons | Skip the LLM refine call entirely. |
| LLM merges two candidates of incompatible types | Post-check: if a cluster mixes types, split it and log ERROR. |
| Same name, genuinely different entities | Refine splits them by evidence → two `ResolvedEntity` with the same `canonical_name` but **different ids**. Relations referencing that bare name map by the name→id table; when a name maps to >1 id (the split), attach to the more frequent one and log WARNING. (Endpoint-by-id keeps everything else unambiguous.) |
| Relation endpoint name not in any entity | Log WARNING, drop the relation. |
| Same `(head_id, type, tail_id)` + overlapping evidence | Collapse into one; keep both evidence; `confidence` = max; attributes from the higher-confidence instance. |
| Same `(head_id, type, tail_id)` + different evidence | Keep separate — distinct instances. |
| Empty `entities` and `relations` | Return both empty; no LLM call. |

---

## API (`api/routers/resolve.py`)

```python
from fastapi import APIRouter, HTTPException
import httpx

from api.models import ResolveRequest, ResolveResponse
from api.warmth import mark_warm
from nlp.resolver import service as resolver_service

router = APIRouter()


@router.post("/resolve", response_model=ResolveResponse)
def resolve(req: ResolveRequest) -> ResolveResponse:
    try:
        result = resolver_service.resolve(req.entities, req.relations)
    except httpx.HTTPError:
        raise HTTPException(503, "ollama_unavailable")
    mark_warm("resolve")
    return result
```

Register in `api/main.py`. In-process orchestrators may call `resolver_service.resolve(...)`
directly. Models go in `api/models.py`.

---

## Configuration

| Env var | Default | Description |
|---|---|---|
| `RESOLVE_MODEL` | `qwen2.5:32b` | Ollama model for the refine call |
| `RESOLVE_TIMEOUT` | `180` | Seconds per Ollama call |
| `RESOLVE_CANDIDATE_BATCH` | `80` | Max candidates per refine call before batching |
| `OLLAMA_HOST` | `http://ollama:11434` | Shared |

---

## Dependencies

- Ollama sidecar (refine call).
- `vectorizer` (only if pre-merge blocks by embedding similarity — spec §3.7).
- `httpx` — already in requirements.
- No new packages.
