# Entity Extractor — Design & Implementation

## Purpose

A **stateless extraction function**: given a passage of text and a schema describing what to
look for, return the entities and relations found in that text, structured as guaranteed-valid
JSON.

The service knows nothing about documents, chunks, character offsets, use cases, or where it
sits in any pipeline. It does not load configuration. Everything it needs arrives in the
request: the **text** and the **schema**. The caller (the orchestrator) owns chunking,
provenance, IDs, offset bookkeeping, and what to do with the result.

```
            text + schema
                 │
                 ▼
        ┌──────────────────┐
        │  Entity Extractor │   build grammar + prompt from schema →
        │   (pure function) │   constrain LLM → check rules → return
        └──────────────────┘
                 │
                 ▼
        entities[] + relations[]
```

---

## File layout

```
nlp/
  entity_extractor/
    __init__.py
    service.py          ← build prompt + grammar from schema, call Ollama, check rules, return
    schema_compiler.py  ← schema → (JSON-Schema grammar, prompt text). Pure, cached by schema hash.
    coref.py            ← resolve_pronouns(text) → text  (document-level preprocessing, see below)
    prompts/
      extraction.txt    ← static instruction text; the type catalogue is injected into it
      coref.txt         ← pronoun-resolution rewrite prompt
    ollama_client.py    ← thin /api/chat wrapper with format= JSON Schema

api/
  routers/
    entity_extract.py   ← POST /entity-extract  (thin, stateless wrapper over service.run)
```

---

## Preprocessing — pronoun resolution

Extraction quality, especially for relations, suffers when a subject or object is a bare
pronoun: *"Fue nombrada ministra. Ella transfirió 4,2 M€ a Delta S.A."* — the payment's subject
is "Ella", which we tell the extractor to skip, so the relation is lost or unanchored.

So before extraction we **rewrite pronouns to the names they refer to**:
*"…Laura Méndez transfirió 4,2 M€ a Delta S.A."* Now the extractor sees an explicit named
subject and captures the relation cleanly.

**This runs on the whole document, before chunking — not inside the per-chunk extract call.**
A pronoun's antecedent is frequently in an earlier sentence (often an earlier chunk), so
resolution needs full-document context. Doing it per chunk would miss exactly those cases and
would break the extractor's statelessness. `coref.resolve_pronouns(document_text)` is therefore
a document-level helper the orchestrator calls **upstream of the chunker**; the chunks handed to
`/entity-extract` already carry explicit names, and the endpoint itself stays a pure
`text + schema` function.

```
full document → coref.resolve_pronouns() → chunker → /entity-extract (per chunk)
```

Implementation: an LLM rewrite pass (over the document, windowed if long) that replaces third-
person pronouns with their antecedent's most complete name and otherwise returns the text
**unchanged** — same language (Spanish), no paraphrasing, no added or dropped content. qwen
handles Spanish coreference well; a classical coref model (spaCy + coreferee / fastcoref) is the
non-LLM alternative but has weaker Spanish support. The extractor's "do not extract bare
pronouns" rule remains as a safety net for any pronoun the rewrite leaves behind.

---

## Data contracts

### Input — `ExtractRequest`

Two fields. Nothing else.

```python
class ExtractRequest(BaseModel):
    text:   str
    schema: ExtractionSchema
```

#### The schema (passed inline)

The caller describes the types directly — no file, no registry. Datatypes are limited to
`string`, `number`, `boolean`.

```python
class AttributeDef(BaseModel):
    name:        str
    datatype:    Literal["string", "number", "boolean"]
    description: str = ""

class SubtypeDef(BaseModel):
    name:        str
    description: str = ""

class EntityTypeDef(BaseModel):
    name:        str
    description: str = ""
    subtypes:    list[SubtypeDef]   = []
    attributes:  list[AttributeDef] = []

class RelationTypeDef(BaseModel):
    name:        str
    description: str = ""
    subtypes:    list[SubtypeDef]   = []
    attributes:  list[AttributeDef] = []
    head_types:  list[str]          # allowed subject entity-type names
    tail_types:  list[str]          # allowed object entity-type names

class ExtractionSchema(BaseModel):
    entity_types:   list[EntityTypeDef]
    relation_types: list[RelationTypeDef]
```

That is the complete set of knobs: entity types (name + description), their subtypes, their
per-type attributes (name + datatype); relation types (name + description), their subtypes,
their per-type attributes, and the entity types allowed as subject (`head_types`) and object
(`tail_types`).

### Output — `ExtractResponse`

```python
class ExtractedEntity(BaseModel):
    name:          str           # most complete form of the entity in the text
    mention_text:  str           # verbatim substring the model copied (this mention)
    type:          str           # one of the schema's entity types
    subtype:       str | None
    evidence_text: str           # verbatim sentence the mention sits in (context for clustering)
    attributes:    dict[str, str | float | bool]   # only this type's attributes, correct datatypes
    confidence:    float          # the model's own 0–1 estimate (unprocessed)

class ExtractedRelation(BaseModel):
    head:          int           # index into entities[]  (subject)
    tail:          int           # index into entities[]  (object)
    type:          str           # one of the schema's relation types
    subtype:       str | None
    evidence_text: str           # verbatim clause/sentence stating the relation
    attributes:    dict[str, str | float | bool]
    confidence:    float

class ExtractResponse(BaseModel):
    entities:  list[ExtractedEntity]
    relations: list[ExtractedRelation]
```

No spans, no offsets, no echoed IDs. The model copies `mention_text` / `evidence_text`
verbatim; if the caller wants character positions it locates those strings in its own text —
the extractor doesn't, because it doesn't know what "the document" is.

---

## Algorithm

### 1. Compile the schema → grammar + prompt  (`schema_compiler.py`, pure, cached)

Both artefacts are derived from `request.schema` and cached by a hash of it (the same schema
is reused across many calls, so this is built once).

**Grammar** — a JSON Schema that Ollama turns into a decoding constraint. It is a
**discriminated union**: one branch per entity type and one per relation type. Each entity
branch pins `type` to a constant, lists *only that type's* subtypes (as an enum) and *only
that type's* attributes (each typed `string→string`, `number→number`, `boolean→boolean`). A
leading `reasoning` string field comes first. This is what makes three of the four rules
**impossible to violate** — the model literally cannot emit a wrong subtype, a foreign
attribute, or a wrong datatype.

```jsonc
{
  "type": "object",
  "required": ["reasoning", "entities", "relations"],
  "properties": {
    "reasoning": {"type": "string"},
    "entities": { "type": "array", "items": { "oneOf": [ /* one branch per entity type */ ] }},
    "relations":{ "type": "array", "items": { "oneOf": [ /* one branch per relation type */ ] }}
  }
}
// entity branch: type=const, subtype∈{its subtypes}|null,
//   attributes={its attributes, each correct datatype}, name, mention_text, evidence_text, confidence
// relation branch: head:int, tail:int, type=const, subtype, evidence_text, attributes, confidence
```

**Prompt** — the static `extraction.txt` instructions with the schema's type catalogue
(type/subtype/attribute names + descriptions) injected.

### 2. Call Ollama

```python
raw = ollama_client.extract(system=prompt, user=text, grammar=grammar)
# temperature=0, num_ctx sized to fit prompt + text; reasoning is decoded then discarded
```

### 3. Check rules (the only post-processing)

The grammar already guarantees valid subtypes, attributes, and datatypes. That leaves exactly
one thing the grammar cannot express — it can't see which entity type sits at a given index:

| Rule | Action on violation |
|---|---|
| relation `head`/`tail` is a valid index into `entities[]` | drop the relation, log WARNING |
| indexed `head` entity's type ∈ relation's `head_types`, and `tail` ∈ `tail_types` | drop the relation, log WARNING |

That's it. Entities are returned as-is. No confidence penalties, no review flags, no required-
attribute logic — the schema constrains structure, these two checks enforce the one relational
rule, and anything malformed beyond that simply can't be produced.

### 4. Return `ExtractResponse`.

---

## Prompt template (`extraction.txt`)

```
ROLE
====
You are an expert information-extraction system. You read a passage of text and return every
entity and every relation it explicitly states or directly implies, as JSON conforming to the
schema. Never invent information that is not in the text.

Fill the `reasoning` field FIRST: in one or two sentences, note which entities and relations
the text contains. Then fill `entities` and `relations`. Do all thinking in `reasoning` —
you cannot write anything outside the JSON.

TYPES
=====
{type_catalogue}
(Only the types listed here exist. If something matches no type, do not extract it.)

WHAT TO EXTRACT
===============
- Entities: every distinct real-world entity that matches a type above — named mentions
  (a full personal or organisation name) and definite descriptions that identify a specific
  entity ("the finance minister"). Do NOT extract bare pronouns ("he", "it") or generic nouns
  ("a company", "investors", "money").
- Relations: only those whose subject AND object are entities you listed, and which the text
  states or directly implies. Never infer relations from world knowledge.

FIELDS
======
- name: the most complete form of the entity in the text. Preserve original language/spelling.
- mention_text: copy the exact surface string for this mention, character-for-character (may
  differ from name). It MUST be a verbatim substring of the text.
- head / tail: the INTEGER POSITIONS (0-based) of the subject and object entities in the
  `entities` array you are producing — not names.
- evidence_text: copy verbatim — for an entity, the sentence the mention appears in; for a
  relation, the clause that states it.
- subtype: use only when the text clearly supports it; otherwise null.
- attributes: fill one only when its value is explicitly stated; never guess; leave unstated
  attributes out. Normalise dates/codes only when the conversion is certain.
- confidence: your own 0–1 estimate. 0.9–1.0 explicit, 0.6–0.8 implied, <0.3 don't extract.
```

---

## Ollama client (`entity_extractor/ollama_client.py`)

Separate from the summarizer's client so models/timeouts are independent.

```python
EXTRACTION_MODEL   = os.environ.get("EXTRACTION_MODEL",   "qwen2.5:32b")
EXTRACTION_TIMEOUT = float(os.environ.get("EXTRACTION_TIMEOUT", "180"))
EXTRACTION_NUM_CTX = int(os.environ.get("EXTRACTION_NUM_CTX", "8192"))

def extract(system: str, user: str, grammar: dict) -> dict:
    response = httpx.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model":    EXTRACTION_MODEL,
            "messages": [{"role": "system", "content": system},
                         {"role": "user",   "content": user}],
            "stream":   False,
            "format":   grammar,
            "options":  {"temperature": 0, "num_ctx": EXTRACTION_NUM_CTX},
        },
        timeout=EXTRACTION_TIMEOUT,
    )
    response.raise_for_status()
    return json.loads(response.json()["message"]["content"])
```

Retry: up to 3 attempts, backoff `2 ** attempt`, on `httpx.HTTPError` / `json.JSONDecodeError`
/ `KeyError`. `num_ctx` must hold prompt + text or Ollama **silently truncates** — size it to
the schema (8192 is ample for a handful of types; raise it for large schemas).

---

## API router (`api/routers/entity_extract.py`)

A thin stateless wrapper — no state, no config load.

```python
@router.post("/entity-extract", response_model=ExtractResponse)
def entity_extract(req: ExtractRequest) -> ExtractResponse:
    if not req.text.strip():
        raise HTTPException(422, "text must be non-empty")
    try:
        return entity_extractor_service.run(req.text, req.schema)
    except httpx.HTTPError:
        raise HTTPException(503, "ollama_unavailable")
```

(If the orchestrator is in-process Python it can skip the endpoint and call
`entity_extractor_service.run(text, schema)` directly — there is no hidden state either way.)

---

## Configuration

| Env var | Default | Description |
|---|---|---|
| `EXTRACTION_MODEL` | `qwen2.5:32b` | Ollama model tag for extraction |
| `EXTRACTION_TIMEOUT` | `180` | Seconds per Ollama call |
| `EXTRACTION_NUM_CTX` | `8192` | Context window; must hold prompt + text |
| `OLLAMA_HOST` | `http://ollama:11434` | Shared with summarizer |

---

## Testing

See [06-testing.md](06-testing.md) §4. Pure-Python, no LLM: `schema_compiler` produces the
right discriminated-union grammar and prompt from a schema; the rule check drops relations
with out-of-range or type-incompatible endpoints and keeps everything else. One mocked-Ollama
integration test exercises `run()` end-to-end.

---

## Dependencies

- Ollama sidecar — same container already used by the summarizer
- `httpx` — already in requirements.txt
- No config loader, no `nlp/schema.py` on the extraction path, no new Python packages
