# Intra-Document Entity Clustering — Design & Implementation

## Purpose

After all chunks of a document have been processed by the Entity Extractor, this step
clusters entity mentions that refer to the same real-world entity within the document and
resolves pronouns and role references to the named entities already found.

The output is a flat list of `LocalEntity` objects — one per canonical entity within the
document — each carrying its surface mentions, absolute offsets, and a provisional
description. These become the inputs to cross-document disambiguation.

---

## Position in pipeline

```
Entity Extractor (×N chunks)
    │  entities[], relations[] per chunk
    ▼
[Intra-Document Entity Clustering]   POST /cluster
    │  LocalEntity[], edges[]
    ▼
Entity Disambiguator  (per LocalEntity)
```

---

## What this step does and does not do

| Does | Does not |
|---|---|
| Cluster surface mentions to a canonical name + `local_id` | Deduplicate edges between the same pair |
| Resolve pronouns and roles ("the CEO", "he") to named entities | Merge or sum relation attributes |
| Resolve relation endpoints from surface strings to `local_id`s | Contact any external index or store |
| Produce absolute character offsets for every mention | Make any cross-document decisions |

Multiple edges between the same `(head_local_id, relation, tail_local_id)` triple are kept
as separate events. If the document says a payment happened twice, that is two edges. The
downstream graph handles it.

---

## File layout

```
nlp/
  clusterer/
    __init__.py
    service.py          ← orchestration: clustering call + pronoun resolution + endpoint wiring
    prompts/
      cluster.txt       ← system prompt for entity clustering
      resolve.txt       ← system prompt for pronoun / role resolution
    ollama_client.py    ← /api/chat wrapper (same pattern as entity_extractor/ollama_client.py)

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
    text:       str           # original chunk text (needed for pronoun context)
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
    text:       str     # surface form as extracted
    chunk_id:   str
    char_start: int     # absolute offset (chunk.char_start + span.start)
    char_end:   int

class LocalEntity(BaseModel):
    local_id:       str           # e.g. "e001", scoped to this document
    canonical_name: str
    type:           str
    subtype:        str | None
    description:    str           # from the LLM clustering call
    mentions:       list[EntityMention]
    attributes:     dict[str, Any] = {}   # union of non-null attributes from mentions

class LocalEdge(BaseModel):
    head_local_id:  str
    relation:       str
    tail_local_id:  str
    description:    str
    attributes:     dict[str, Any] = {}
    evidence_span:  AbsoluteSpan          # absolute offsets

class AbsoluteSpan(BaseModel):
    chunk_id:   str
    char_start: int
    char_end:   int
    text:       str               # evidence sentence(s)

class ClusterResponse(BaseModel):
    doc_id:         str
    local_entities: list[LocalEntity]
    edges:          list[LocalEdge]
```

---

## Algorithm

### Step 1 — Flatten all mentions (pure Python)

Collect every `ExtractedEntity` from every chunk into a numbered flat list. Compute
absolute offsets (`chunk.char_start + span.start/end`) for each mention.

### Step 2 — Entity clustering (LLM call)

Send the flat mention list to the LLM with the `cluster.txt` prompt. The LLM assigns each
mention to a cluster by integer index (not by repeating name strings — indices are
unambiguous and token-efficient).

**Output schema:**

```jsonc
{
  "type": "object",
  "required": ["clusters"],
  "properties": {
    "clusters": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["cluster_id", "canonical_name", "type", "subtype", "description", "mention_indices"],
        "additionalProperties": false,
        "properties": {
          "cluster_id":      {"type": "string"},
          "canonical_name":  {"type": "string"},
          "type":            {"enum": ["PERSON", "ORGANIZATION", ...]},
          "subtype":         {"oneOf": [{"enum": [...]}, {"type": "null"}]},
          "description":     {"type": "string"},
          "mention_indices": {"type": "array", "items": {"type": "integer"}}
        }
      }
    }
  }
}
```

Each entry in `mention_indices` is the 0-based position in the flat mention list.

**Assign `local_id`**: after the LLM returns, assign `e001`, `e002`, … in cluster order
(pure Python, stable within this document).

**Validation**: every mention index must appear in exactly one cluster. Unclustered indices
are logged at WARNING and placed in singleton clusters.

**Batching**: if the flat mention list exceeds `CLUSTER_ENTITY_BATCH` (default 80), split
by type and run one clustering call per type batch. After all calls, merge clusters that
share the same `canonical_name` across batches (case-insensitive).

### Step 3 — Pronoun and role resolution (LLM call, optional)

If the document contains entity mentions whose `type` was not determinable or whose surface
form is a pronoun ("he", "she", "they") or a pure role title ("the minister", "the CEO"),
send a second constrained call with the `resolve.txt` prompt.

Input: the list of unresolved mentions + the `LocalEntity` list produced in Step 2 as
resolution candidates.

Output schema:

```jsonc
{
  "type": "object",
  "required": ["resolutions"],
  "properties": {
    "resolutions": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["mention_index", "resolved_local_id"],
        "additionalProperties": false,
        "properties": {
          "mention_index":     {"type": "integer"},
          "resolved_local_id": {"oneOf": [{"type": "string"}, {"type": "null"}]}
        }
      }
    }
  }
}
```

`null` means the LLM could not resolve — those mentions stay unlinked and are excluded from
edges but logged at DEBUG.

Step 3 is skipped entirely if no unresolved mentions are detected (saves one LLM call for
the common case).

### Step 4 — Endpoint resolution (pure Python)

For each `ExtractedRelation`, look up `head` and `tail` (surface strings) in the mention
→ `local_id` map produced in Steps 2–3:

1. Exact string match first.
2. If no match, normalise (lowercase, strip punctuation) and retry.
3. If still no match, log WARNING and set `head_local_id` / `tail_local_id = null`; keep the
   edge (flagged for downstream review via low confidence).

Absolute evidence offsets are computed from `chunk.char_start + relation.evidence_span.*`.

### Step 5 — Return `ClusterResponse`

No further merging or deduplication. Multiple edges with the same `(head_local_id, relation,
tail_local_id)` are kept as-is — they represent distinct events.

---

## Prompt templates

### `cluster.txt`

```
You are resolving entity mentions to canonical identities within a single document.
You will receive a numbered list of entity mentions with their types and surrounding context.
Group mentions that refer to the same real-world entity into clusters.

ENTITY TYPES
============
{ontology_entity_types}

RULES
=====
- Assign every mention to exactly one cluster.
- Choose the most complete, unambiguous name as canonical_name (full name over nickname,
  official name over alias).
- Keep type consistent within a cluster. If a mention was mis-typed, use the majority type.
- Write a concise single-sentence description covering all mentions in the cluster.
- Do not merge entities of different broad types (never merge a PERSON into an ORGANIZATION).
- Use the integer index (0-based) in mention_indices — do not repeat name strings.

MENTIONS
========
{mention_list}
```

### `resolve.txt`

```
You are resolving pronouns and role references to named entities already identified in
this document.

KNOWN ENTITIES
==============
{entity_list}

UNRESOLVED MENTIONS
===================
{unresolved_list}

RULES
=====
- For each unresolved mention, output its index and the local_id of the entity it refers to.
- Set resolved_local_id to null if you cannot determine the referent with confidence.
- Do not invent new entities — only resolve to entities in the KNOWN ENTITIES list.
```

---

## Configuration

| Env var | Default | Description |
|---|---|---|
| `CLUSTER_MODEL` | `gemma4:31b` | Ollama model for clustering and resolution |
| `CLUSTER_TIMEOUT` | `120` | Seconds per Ollama call |
| `CLUSTER_ENTITY_BATCH` | `80` | Max mentions per clustering call before batching |
| `OLLAMA_HOST` | `http://ollama:11434` | Shared |

---

## Testing

See [06-testing.md](06-testing.md) §2 — Steps 1, 4, and 5 are pure Python and fully
unit-testable without mocking. LLM clustering (Step 2) and resolution (Step 3) are tested
with a mocked Ollama response. Key cases: overlapping spans, unclustered mention fallback,
no-pronoun fast path (Step 3 skipped), unresolved endpoint fallback.

---

## Dependencies

- `nlp/schema.py` — entity type enum for output schema and prompts
- Ollama sidecar
- `httpx` — already in requirements.txt
- No new packages
