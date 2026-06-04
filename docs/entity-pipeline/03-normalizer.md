# Intra-Document Entity Clustering — Design & Implementation

## Purpose

After all chunks of a document have been processed by the Entity Extractor, this step
clusters entity mentions that refer to the same real-world entity within the document
and wires relation endpoints to cluster `local_id`s — all in a single LLM call.

This step is relatively lightweight for typical journalism documents (10–50 entities per
document). Its main value is producing stable `local_id`s so that cross-document
disambiguation has clean, canonical inputs rather than raw surface strings.

---

## Position in pipeline

```
Entity Extractor (×N chunks)
    │  entities[], relations[] per chunk
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
| Assign `head_local_id` / `tail_local_id` to every edge (LLM) | Merge or sum relation attributes |
| Compute absolute char offsets for every mention | Resolve pronouns or unidentifiable role titles |
| Annotate likely-same pairs via Jaccard before the LLM call | Contact any external index or store |

Multiple edges with the same `(head_local_id, relation, tail_local_id)` triple are kept
as separate events.

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
    char_start: int
    char_end:   int
    text:       str
    entities:   list[ExtractedEntity]
    relations:  list[ExtractedRelation]

class ClusterRequest(BaseModel):
    doc_id:   str
    use_case: str
    chunks:   list[ChunkExtraction]
```

### Response — `ClusterResponse`

```python
class EntityMention(BaseModel):
    text:       str
    chunk_id:   str
    char_start: int     # absolute offset
    char_end:   int

class LocalEntity(BaseModel):
    local_id:       str           # "e001", scoped to this document
    canonical_name: str
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

Collect every `ExtractedEntity` from every chunk into a numbered flat list (`mention_index`,
0-based). Compute absolute offsets for each mention. Collect every `ExtractedRelation` into
a separate numbered flat list (`relation_index`).

### Step 2 — Jaccard hints (pure Python)

For every pair of mentions, compute Jaccard similarity on word tokens of their names. Pairs
with score ≥ `JACCARD_HINT_THRESHOLD` (default 0.5) are annotated `LIKELY_SAME` in the
mention list sent to the LLM. This is O(n²) on token sets — negligible for typical document
sizes — and gives the LLM focused hints without requiring it to compare every pair itself.

### Step 3 — Combined LLM call

A single call clusters all mentions AND assigns `local_id`s to relation endpoints. Sending
both tasks together means the LLM uses the cluster assignments it just defined when it wires
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

### Step 4 — Post-process (pure Python)

1. Every mention index must appear in exactly one cluster. Unclustered indices → WARNING +
   singleton cluster.
2. For each edge, recover `attributes` and `evidence_span` (absolute offsets) by looking up
   `relation_index` in the flat relation list.
3. Validate `local_id` uniqueness within the document; suffix duplicates `_b`, `_c`.

**Batching:** if the flat mention list exceeds `CLUSTER_ENTITY_BATCH` (default 80), split
by type, run one clustering call per batch, then merge clusters sharing the same
`canonical_name` (case-insensitive) across batches. Run one final edge-resolution call
over the merged cluster list.

---

## Prompt template (`cluster.txt`)

```
You are resolving entity mentions to canonical identities within a single document,
and wiring extracted relations to those identities.

ENTITY TYPES
============
{ontology_entity_types}

MENTIONS  (0-based index, type, name, [annotation])
====================================================
{mention_list}

RELATIONS  (0-based index, head name, relation, tail name)
==========================================================
{relation_list}

RULES — CLUSTERS
================
- Assign every mention to exactly one cluster.
- Choose the most complete, unambiguous name as canonical_name.
- Mentions annotated LIKELY_SAME are strong candidates to share a cluster — verify with context.
- Do not merge entities of incompatible types.
- Write a concise single-sentence description for each cluster.

RULES — EDGES
=============
- For each relation, set head_local_id and tail_local_id to the local_id of the matching cluster.
- If an endpoint cannot be matched to any cluster, set it to null.
- Write a concise description for each edge.
```

---

## Configuration

| Env var | Default | Description |
|---|---|---|
| `CLUSTER_MODEL` | `gemma4:31b` | Ollama model for clustering |
| `CLUSTER_TIMEOUT` | `120` | Seconds per Ollama call |
| `CLUSTER_ENTITY_BATCH` | `80` | Max mentions per call before batching |
| `JACCARD_HINT_THRESHOLD` | `0.5` | Score above which to annotate LIKELY_SAME |
| `OLLAMA_HOST` | `http://ollama:11434` | Shared |

---

## Testing

See [06-testing.md](06-testing.md) §2 — Steps 1, 2, and 4 are pure Python with no LLM or
network dependency. Step 3 is tested with mocked Ollama responses. Key cases: Jaccard hint
triggers LIKELY_SAME annotation, unclustered mention fallback, edge with null endpoint
retained, batching merge by canonical_name.

---

## Dependencies

- `nlp/schema.py` — entity type enum for output schema and prompts
- Ollama sidecar
- `httpx` — already in requirements.txt
- No new packages
