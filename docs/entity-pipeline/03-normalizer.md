# Entity Resolver — Design & Implementation

*(Formerly "the normalizer". Intra-document entity + relation resolution.)*

## Purpose

A **stateless resolution service**: given a document's entity mentions **and the relations
between them**, reduce the mentions to canonical entities and rewrite the relations to use those
canonical names.

It is intra-document by nature — the caller invokes it once per document — so there is **no
`doc_id`** and nothing about chunks, offsets, use-case, or schema. Entities and relations
reference each other by **name**.

```
  IN   entities:  [{name, type, evidence}, ...]
       relations: [{head:name, tail:name, type, evidence, attributes}, ...]
                       │
                       ▼
              ┌────────────────────┐
              │   Entity Resolver   │  pre-merge → LLM refine → rewrite relations
              └────────────────────┘
                       │
                       ▼
  OUT  entities:  [{canonical_name, names[], type, evidence[]}, ...]   ← reduced
       relations: [{head:canonical_name, tail:canonical_name, type, evidence[], attributes}, ...]
```

---

## File layout

```
nlp/
  resolver/
    __init__.py
    service.py        ← pre-merge, LLM refine, rewrite relations, return
    premerge.py       ← pure-Python exact-name (or embedding) blocking
    edges.py          ← pure-Python relation rewrite + overlap collapse (no LLM)
    prompts/
      resolution.txt  ← the refine prompt
    ollama_client.py  ← /api/chat with the cluster schema

api/
  routers/
    resolve.py        ← POST /resolve
```

---

## Data contracts

### Input — `POST /resolve`

```python
class EntityIn(BaseModel):
    name:     str            # the surface name as extracted ("Laura Méndez", "Méndez", "la ministra")
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
to its canonical entity. (The orchestrator produces this by flattening the chunks' extractions
and converting the extractor's relation indices into the endpoint entity's name.)

### Output — `ResolveResponse`

```python
class ResolvedEntity(BaseModel):
    canonical_name: str            # the chosen name for the reduced entity
    names:          list[str]      # every surface name merged into it (its aliases)
    type:           str
    subtype:        str | None
    evidence:       list[str]      # every member's evidence sentence, collected

class ResolvedRelation(BaseModel):
    head:       str                # a canonical_name
    tail:       str                # a canonical_name
    type:       str
    subtype:    str | None
    evidence:   list[str]          # collected; >1 only when overlap duplicates collapsed
    attributes: dict               # passed through, never aggregated
    confidence: float              # max across collapsed duplicates

class ResolveResponse(BaseModel):
    entities:  list[ResolvedEntity]
    relations: list[ResolvedRelation]
```

The reduced entity keeps both its `canonical_name` and the `names` it absorbed, so the caller
can match anything back. Relations now read in canonical names end to end.

---

## Algorithm

### Step 1 — Pre-merge entities (pure Python, no LLM)

Normalize each entity's name (lowercase; strip honorifics, punctuation, legal suffixes like
"S.A." / "Ltd") and group entities that share a normalized name **and** type into one
**candidate**. Most entries are literal repeats ("Laura Méndez" / "Méndez" 20×), so this
collapses a long list into a handful of candidates, each keeping its member names, a provisional
canonical name (most complete form), and one representative evidence sentence.

> Optionally block by `bge-m3` embedding similarity instead of exact form, to catch spelling
> variants ("Banco Santander" / "Santander S.A."). Reuses the disambiguator's encoder (04).

### Step 2 — LLM refine (the only LLM call)

Send the **candidates** (not the raw entities) to the LLM, each with its provisional name, type,
and representative evidence. It does only what pre-merge cannot: merge a definite description or
pronoun mention into a named candidate, and split same-named different entities when their
evidence diverges. It references candidates by integer index.

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

**Skip the LLM entirely** when pre-merge yields all unique-named singletons. If candidates
exceed `RESOLVE_CANDIDATE_BATCH`, refine per type then merge across types only where names
match. The LLM input scales with *candidate* count, not entity count — so it stays small even
for very long inputs.

### Step 3 — Build reduced entities + the name map (pure Python)

For each refined cluster, emit a `ResolvedEntity` (canonical_name, all member `names`, type,
collected `evidence`). Build `name → canonical_name` from the members — the lookup Step 4 uses.

### Step 4 — Rewrite and de-duplicate relations (pure Python, `edges.py`)

Deterministic, no LLM:

1. **Rewrite endpoints:** map each relation's `head`/`tail` name to its `canonical_name` via the
   Step 3 map. A name in no cluster → log WARNING, drop the relation.
2. **Collapse overlap duplicates:** group by `(head, type, tail, evidence_fingerprint)`;
   relations with the same canonical endpoints **and** overlapping evidence (≥ 80 %, the same
   sentence from two overlapping chunks) collapse into one — keeping both evidence snippets,
   `confidence` = max, attributes from the highest-confidence instance.
3. **Distinct evidence = distinct relations.** Two payments between the same parties are two
   payments. Attributes are **never** aggregated.

---

## Prompt template (`resolution.txt`)

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
- Do NOT write a description — that is produced later.

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
| Same name, genuinely different entities | The refine step splits them by evidence → two `ResolvedEntity` with **the same `canonical_name`**. Relations referencing that bare name are then ambiguous — log WARNING and attach to the more frequent one. (The only case a name can't disambiguate; rare intra-doc.) |
| Relation endpoint name not in any entity | Log WARNING, drop the relation. |
| Same `(head, type, tail)` + overlapping evidence | Collapse into one relation; keep both evidence; `confidence` = max; attributes from the higher-confidence instance. |
| Same `(head, type, tail)` + different evidence | Keep separate — distinct instances. |
| Empty `entities` and `relations` | Return both empty; no LLM call. |

---

## Configuration

| Env var | Default | Description |
|---|---|---|
| `RESOLVE_MODEL` | `qwen2.5:32b` | Ollama model for the refine call |
| `RESOLVE_TIMEOUT` | `180` | Seconds per Ollama call |
| `RESOLVE_CANDIDATE_BATCH` | `80` | Max candidates per refine call before batching |
| `OLLAMA_HOST` | `http://ollama:11434` | Shared |

---

## Testing

See [06-testing.md](06-testing.md) §2 — `premerge.py` and `edges.py` are fully unit-testable
(no LLM); the LLM refine is tested with a mocked Ollama response. Key cases: pre-merge collapses
exact-name repeats, refine merges a definite-description candidate, refine splits same-name
different entities, all-singletons skips the LLM, relation rewrite maps names to canonical,
overlap collapse, distinct evidence stays separate, dangling endpoint dropped.

---

## Dependencies

- Ollama sidecar
- `bge-m3` encoder (only if pre-merge blocks by embedding similarity; reuses 04's)
- `httpx` — already in requirements.txt
- No new packages
