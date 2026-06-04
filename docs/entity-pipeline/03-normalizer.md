# Normalizer — Design & Implementation

## Purpose

Pass 2 of the document-level extraction pipeline. Operates once per document after all
chunks have been extracted. It performs two operations in a single LLM call:

1. **Intra-document entity resolution** — cluster all entity mentions from all chunks into
   canonical local entities (`local_id`). "Musk", "Elon Musk", and "the billionaire" become
   one cluster with canonical name "Elon Musk".

2. **Edge deduplication** — when multiple chunks independently extract the same
   `(head, relation_type, tail)` triple, merge them into a single edge, accumulating all
   evidence spans and merging attributes (sum amounts, union dates).

Relation type normalization is **not** a task of this module — canonical relation types are
already assigned in Pass 1 via enum-constrained decoding. This module works with those
canonical types from the start.

After this module, relation endpoints are `local_id`s, not surface name strings. Cross-document
disambiguation (mapping `local_id → canonical_id` in the graph) is a separate downstream step.

---

## Position in pipeline

```
Chunker  →  Entity Extractor (×N chunks)  →  [Normalizer]  →  Disambiguator  →  Persistence
                                               (per doc, Pass 2)
```

---

## File layout

```
nlp/
  normalizer/
    __init__.py
    service.py            ← orchestration: build prompt, call LLM, post-process, return
    merger.py             ← pure-Python edge merging logic (no LLM)
    prompts/
      resolution.txt      ← system prompt for entity clustering
    ollama_client.py      ← calls /api/chat with resolution schema

api/
  routers/
    normalize.py          ← POST /normalize
```

---

## Data contracts

### Request — `POST /normalize`

```python
class ChunkExtraction(BaseModel):
    chunk_id:   str
    char_start: int
    char_end:   int
    entities:   list[ExtractedEntity]    # from EntityExtractResponse
    relations:  list[ExtractedRelation]  # from EntityExtractResponse

class NormalizeRequest(BaseModel):
    doc_id:   str
    use_case: str
    chunks:   list[ChunkExtraction]      # all chunks for this document, ordered
```

### Response — `NormalizeResponse`

```python
class EntityMention(BaseModel):
    text:       str     # surface form as extracted
    chunk_id:   str
    char_start: int     # absolute offset (chunk.char_start + span.start)
    char_end:   int     # absolute offset (chunk.char_start + span.end)
    confidence: float

class LocalEntity(BaseModel):
    local_id:       str             # e.g. "e001", scoped to this document
    canonical_name: str
    type:           str
    subtype:        str | None
    description:    str             # synthesised from all mention descriptions
    mentions:       list[EntityMention]

class MergedEdge(BaseModel):
    head_local_id:  str
    relation:       str             # canonical type from ontology
    tail_local_id:  str
    description:    str             # synthesised from all relation descriptions
    attributes:     RelationAttributes   # merged (see rules below)
    evidence:       list[EvidenceSpan]   # one per source chunk

class EvidenceSpan(BaseModel):
    chunk_id:   str
    char_start: int     # absolute
    char_end:   int
    text:       str     # the evidence sentence(s)

class NormalizeResponse(BaseModel):
    doc_id:           str
    local_entities:   list[LocalEntity]
    edges:            list[MergedEdge]
```

---

## Algorithm

### Step 1 — Flatten all mentions across chunks

Collect every `ExtractedEntity` from every chunk into a flat list. Each entry retains its
`chunk_id` and absolute offset (computed by adding `chunk.char_start` to the entity's
relative span). This is pure Python, no LLM.

### Step 2 — Entity clustering (LLM call)

Build the resolution prompt: include the flat entity list with types, descriptions, and a
representative context sentence (the text surrounding the span). Ask the LLM to assign a
`cluster_id` to each mention.

The LLM output schema is intentionally minimal to keep constrained decoding fast:

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
          "cluster_id":     {"type": "string"},
          "canonical_name": {"type": "string"},
          "type":           {"enum": [...]},
          "subtype":        {"oneOf": [{"enum": [...]}, {"type": "null"}]},
          "description":    {"type": "string"},
          "mention_indices":{"type": "array", "items": {"type": "integer"}}
        }
      }
    }
  }
}
```

`mention_indices` references positions in the flat mention list passed in the prompt. This
avoids asking the LLM to repeat name strings (error-prone) — it uses integer indices instead.

**Context window management:** If the flat mention list exceeds ~80 entities (uncommon for
journalism documents), split into batches by type and run multiple clustering calls, then
merge clusters that share identical canonical names across batches.

### Step 3 — Build `LocalEntity` objects (pure Python)

For each cluster returned by the LLM:
- Assign `local_id = f"e{i:03d}"` (stable within this document)
- Collect all `EntityMention` objects using `mention_indices`
- Use the LLM-provided `description` as the entity description (it has seen all mentions)

Validate that every input mention appears in exactly one cluster. Unclustered mentions (LLM
omission) are logged at WARNING and added as singleton clusters.

### Step 4 — Edge deduplication (pure Python, `merger.py`)

The LLM is not involved in edge deduplication. This is deterministic:

1. **Endpoint resolution:** Replace each relation's `head`/`tail` name string with a
   `local_id` by matching against `EntityMention.text` for each cluster.

   - Exact match first; if no match, normalise (lowercase, strip punctuation) and retry.
   - If still no match, log WARNING and set `head_local_id = null` (edge is kept but
     flagged for review).

2. **Group by `(head_local_id, relation, tail_local_id)`.**

3. **Merge attributes** per relation type rules:
   - `amount`: sum all non-null values (multiple payments in same document)
   - `currency`: use the value if unanimous; `"MIXED"` otherwise
   - `date`: collect as a sorted list, expose as `date_range: [min, max]`
   - `direction`: unanimous value; `"MIXED"` otherwise

4. **Merge descriptions:** concatenate with `" | "` separator. The summarizer's
   `describe_entity` can refresh this later as evidence accumulates.

5. **Collect evidence spans:** one per source relation, converted to absolute offsets.

---

## Prompt template (`resolution.txt`)

```
You are resolving entity mentions to canonical identities within a single document.
You will receive a numbered list of entity mentions with their types and descriptions.
Group mentions that refer to the same real-world entity into clusters.

RULES
=====
- Assign every mention to exactly one cluster.
- Choose the most complete, unambiguous name as canonical_name (full name over nickname,
  legal name over alias).
- Keep type and subtype consistent within a cluster; if a mention was wrongly typed, use
  the majority type.
- Write a single-sentence description that covers all mentions in the cluster.
- Only cluster mentions of the same broad type (do not merge a PERSON into an ORGANIZATION).
- Use the integer index from the list (0-based) in mention_indices.

MENTIONS
========
{mention_list}
```

---

## Edge cases

| Situation | Handling |
|---|---|
| Document has only one chunk | Skip LLM call if fewer than 2 distinct entity names across the chunk. Return a trivial 1:1 cluster per entity. |
| LLM assigns same `cluster_id` to two different real entities | Post-check: if two mentions in a cluster have incompatible types, split and log ERROR. |
| Relation head or tail names a mention not in any cluster | Log WARNING, mark edge with `unresolved=true`; persist for review. |
| Attribute conflict (e.g. two different amounts for PAYMENT_TO) | Sum them; include both raw values in `evidence`. |
| Empty document (zero chunks) | Return empty `local_entities` and `edges`; no LLM call. |

---

## Configuration

| Env var | Default | Description |
|---|---|---|
| `NORMALIZATION_MODEL` | `qwen2.5:32b` | Ollama model for entity clustering |
| `NORMALIZATION_TIMEOUT` | `180` | Seconds per Ollama call |
| `NORMALIZATION_ENTITY_BATCH` | `80` | Max mentions per clustering call before batching |
| `OLLAMA_HOST` | `http://ollama:11434` | Shared |

---

## Dependencies

- `nlp/ontology.py` — for entity and relation type enums in schema
- Ollama sidecar
- `httpx` — already in requirements.txt
- No new packages
