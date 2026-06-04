# Intra-Document Entity Clustering — Design & Implementation

## Purpose

After all chunks of a document have been processed by the Entity Extractor, this step
clusters entity mentions that refer to the same real-world entity within the document,
resolves `[PRONOUN]` markers to named entities, and wires relation endpoints to cluster
`local_id`s — all in a single LLM call. The output is `LocalEntity` objects and
`LocalEdge` objects ready for cross-document disambiguation.

---

## Position in pipeline

```
Entity Extractor (×N chunks)
    │  entities[], relations[] per chunk — including [PRONOUN] markers
    ▼
[Intra-Document Entity Clustering]   POST /cluster
    │  LocalEntity[], LocalEdge[] with local_ids and absolute offsets
    ▼
Entity Disambiguator  (per LocalEntity)
```

---

## What this step does and does not do

| Does | Does not |
|---|---|
| Cluster surface mentions to a `local_id` + canonical name | Deduplicate edges between the same pair |
| Resolve `[PRONOUN]` / `[PRONOUN]` markers to named entities | Merge or sum relation attributes |
| Assign `head_local_id` / `tail_local_id` to every edge (LLM) | Contact any external index or store |
| Compute absolute char offsets for every mention | Make any cross-document decisions |

Multiple edges with the same `(head_local_id, relation, tail_local_id)` triple are kept
as separate events. If a payment is described twice in the same document, that is two edges.

---

## File layout

```
nlp/
  clusterer/
    __init__.py
    service.py          ← flatten, Jaccard hints, LLM call, post-process
    prompts/
      cluster.txt       ← combined clustering + endpoint resolution prompt
    ollama_client.py    ← /api/chat wrapper

api/
  routers/
    cluster.py          ← POST /cluster
```

---

## Data contracts

### Request — `POST /cluster`

```python
class ChunkExtraction(BaseModel):
    chunk_id:   str
    char_start: int           # absolute offset of chunk in document
    char_end:   int
    text:       str           # original chunk text (needed for [PRONOUN] context)
    entities:   list[ExtractedEntity]
    relations:  list[ExtractedRelation]

class ClusterRequest(BaseModel):
    doc_id:   str
    use_case: str
    chunks:   list[ChunkExtraction]   # all chunks for the document, ordered
```

### Response — `ClusterResponse`

```python
class EntityMention(BaseModel):
    text:       str     # surface form as extracted ("John Smith", "[PRONOUN]", "the minister")
    chunk_id:   str
    char_start: int     # absolute offset
    char_end:   int

class LocalEntity(BaseModel):
    local_id:       str           # "e001", scoped to this document
    canonical_name: str           # never "[PRONOUN]" — always a resolved name
    type:           str
    subtype:        str | None
    description:    str
    mentions:       list[EntityMention]
    attributes:     dict[str, Any] = {}   # union of non-null attributes across mentions

class LocalEdge(BaseModel):
    head_local_id:  str | None    # null if LLM could not resolve
    relation:       str
    tail_local_id:  str | None
    description:    str
    attributes:     dict[str, Any] = {}
    evidence:       AbsoluteSpan

class AbsoluteSpan(BaseModel):
    chunk_id:   str
    char_start: int
    char_end:   int
    text:       str

class ClusterResponse(BaseModel):
    doc_id:         str
    local_entities: list[LocalEntity]
    edges:          list[LocalEdge]
```

---

## Algorithm

### Step 1 — Flatten (pure Python)

Collect every `ExtractedEntity` from every chunk into a numbered flat list (0-based
`mention_index`). Compute absolute offsets. Collect every `ExtractedRelation` into a
separate numbered flat list (`relation_index`).

### Step 2 — Jaccard name hints (pure Python)

For every pair of mention entries, compute Jaccard similarity on word tokens of their
names. If `jaccard(name_a, name_b) >= 0.5`, annotate that pair as `LIKELY_SAME` in the
mention list sent to the LLM. `[PRONOUN]` mentions are never annotated this way (they have
no meaningful tokens to compare); they are flagged as `NEEDS_RESOLUTION` instead.

This takes O(n²) token-set comparisons on the mention list — negligible for the typical
10–80 entities in a journalism document. It gives the LLM focused hints without requiring
it to compare every pair from scratch.

### Step 3 — Combined LLM call

A single call clusters all mentions AND assigns local_ids to relation endpoints. Sending
both tasks together means the LLM sees the cluster assignments it just made when it wires
the edges — no second pass, no rule-based string matching.

**Output schema:**

```jsonc
{
  "type": "object",
  "required": ["clusters", "edges"],
  "properties": {
    "clusters": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["local_id", "canonical_name", "type", "subtype", "description", "mention_indices"],
        "additionalProperties": false,
        "properties": {
          "local_id":        {"type": "string"},
          "canonical_name":  {"type": "string"},
          "type":            {"enum": ["PERSON", "ORGANIZATION", ...]},
          "subtype":         {"oneOf": [{"enum": [...]}, {"type": "null"}]},
          "description":     {"type": "string"},
          "mention_indices": {"type": "array", "items": {"type": "integer"}}
        }
      }
    },
    "edges": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["relation_index", "head_local_id", "tail_local_id", "description"],
        "additionalProperties": false,
        "properties": {
          "relation_index": {"type": "integer"},
          "head_local_id":  {"oneOf": [{"type": "string"}, {"type": "null"}]},
          "tail_local_id":  {"oneOf": [{"type": "string"}, {"type": "null"}]},
          "description":    {"type": "string"}
        }
      }
    }
  }
}
```

`mention_indices` and `relation_index` reference the flat lists from Step 1. The LLM
assigns `local_id` strings — these are then used directly to wire edges. `canonical_name`
must never be `[PRONOUN]`; if the LLM cannot determine the referent it should still pick
the most reasonable name from context, or set `canonical_name = "[UNRESOLVED]"` which is
caught in Step 4.

### Step 4 — Post-process (pure Python)

1. **Validate coverage**: every mention index must appear in exactly one cluster.
   Unclustered indices → WARNING + singleton cluster.

2. **Reject `[UNRESOLVED]` names**: if `canonical_name == "[UNRESOLVED]"`, log WARNING,
   retain entity with `local_id` and all mentions (for human review queue), set
   `confidence = 0.1` on all its edges.

3. **Recover spans and attributes**: for each edge, look up `relation_index` in the flat
   relation list to recover `evidence_span` (converted to absolute offsets) and
   `attributes`.

4. **Assign `local_id`s from LLM output** — they are already strings assigned by the LLM.
   Validate uniqueness within the document; if duplicates exist, suffix with `_b`, `_c`.

**Batching**: if the flat mention list exceeds `CLUSTER_ENTITY_BATCH` (default 80), split
by type and run one clustering call per type batch. After all calls, merge clusters across
batches that share `canonical_name` (case-insensitive). Run one final call for cross-type
edge resolution only (smaller prompt, no clustering).

---

## Prompt template (`cluster.txt`)

```
You are resolving entity mentions to canonical identities within a single document,
and wiring extracted relations to those identities.

ENTITY TYPES
============
{ontology_entity_types}

MENTIONS  (0-based index, type, name, [annotation])
========================================================
{mention_list}

RELATIONS  (0-based index, head name, relation, tail name)
==========================================================
{relation_list}

RULES — CLUSTERS
================
- Assign every mention to exactly one cluster.
- Choose the most complete, unambiguous name as canonical_name.
- Mentions annotated LIKELY_SAME are strong candidates to share a cluster — verify with context.
- Mentions annotated NEEDS_RESOLUTION are [PRONOUN] markers: resolve them to a named entity
  already present in this document using surrounding context. If truly unresolvable, set
  canonical_name to [UNRESOLVED].
- Do not merge entities of incompatible types.
- Write a concise single-sentence description for each cluster.

RULES — EDGES
=============
- For each relation, set head_local_id and tail_local_id to the local_id of the cluster
  each endpoint belongs to (using the clusters you just defined).
- If an endpoint cannot be matched to any cluster, set it to null.
- Write a concise description for each edge drawn from the relation's context.
```

---

## Configuration

| Env var | Default | Description |
|---|---|---|
| `CLUSTER_MODEL` | `gemma4:31b` | Ollama model for clustering |
| `CLUSTER_TIMEOUT` | `120` | Seconds per Ollama call |
| `CLUSTER_ENTITY_BATCH` | `80` | Max mentions per call before batching |
| `JACCARD_HINT_THRESHOLD` | `0.5` | Jaccard score above which to annotate LIKELY_SAME |
| `OLLAMA_HOST` | `http://ollama:11434` | Shared |

---

## Testing

See [06-testing.md](06-testing.md) §2 — Step 1, 2, and 4 are pure Python with no LLM or
network dependency. The combined LLM call (Step 3) is tested with mocked Ollama responses.
Key cases: `[PRONOUN]` → resolved name, `[PRONOUN]` → `[UNRESOLVED]` fallback, Jaccard
hint triggers LIKELY_SAME annotation, unclustered mention fallback, edge with `null`
endpoint retained.

---

## Dependencies

- `nlp/schema.py` — entity type enum for output schema and prompts
- Ollama sidecar
- `httpx` — already in requirements.txt
- No new packages
