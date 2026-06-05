# NER — Implementation Doc

**Module:** `nlp/ner/` · **Status:** New (written from scratch)
**Authoritative source:** `docs/superpowers/specs/2026-06-05-nlp-two-pipelines-design.md` §3.4

> Naming: the module is `ner` (owner's choice). It does **entity + relation + attribute**
> extraction via an LLM — broader than classic NER. It does not collide with
> `nlp/geotagger/ner.py` (different package). The retired TextRank `nlp/extractor/` is unrelated
> and gone (see doc 01).

## Purpose

A **stateless extraction function**: given a passage of text and a schema describing what to
look for, return the entities and relations found, as guaranteed-valid JSON.

The module knows nothing about documents, chunks, character offsets, use cases, IDs, or where it
sits in a pipeline. It loads no configuration. Everything arrives in the request — the **text**
and the **schema**. The orchestrator (doc 08) owns coref-before-chunking, chunking, provenance,
IDs, and what to do with the result.

```
            text + schema
                 │
                 ▼
        ┌──────────────────┐
        │       ner        │   build grammar + prompt from schema →
        │  (pure function) │   constrain LLM → check the one relational rule → return
        └──────────────────┘
                 │
                 ▼
        entities[] + relations[]
```

The module also provides one **document-level** helper, `coref.resolve_pronouns(text)`, that the
orchestrator calls **before chunking** (not inside `extract`).

---

## File layout

```
nlp/
  ner/
    __init__.py
    service.py          ← run(text, schema): compile → call Ollama → check rule → return
    schema_types.py     ← the inline ExtractionSchema Pydantic contract (shared with the orchestrator)
    schema_compiler.py  ← ExtractionSchema → (JSON-Schema grammar, prompt text). Pure, cached.
    coref.py            ← resolve_pronouns(document_text) → document_text  (document-level)
    ollama_client.py    ← thin /api/chat wrapper with format = JSON-Schema grammar
    prompts/
      extraction.txt    ← static instructions; the type catalogue is injected
      coref.txt         ← pronoun-resolution rewrite prompt
api/
  routers/
    ner.py              ← POST /ner  (thin, stateless wrapper over service.run)
```

---

## The schema (inline, orchestrator-owned)

There is no ontology/schema module in the NLP service. The orchestrator owns the type catalogue
(it loads its own YAML — doc 08) and passes a projected `ExtractionSchema` inline on every call.
The **Pydantic contract** lives here, in `schema_types.py`, because it is the interface `ner`
consumes; the orchestrator imports the same types to build instances.

```python
# nlp/ner/schema_types.py
from pydantic import BaseModel
from typing import Literal

class AttributeDef(BaseModel):
    name:        str
    datatype:    Literal["string", "number", "boolean"]
    description: str = ""

class SubtypeDef(BaseModel):
    name:        str
    description: str = ""

class EntityTypeDef(BaseModel):
    name:       str
    description: str = ""
    subtypes:   list[SubtypeDef]   = []
    attributes: list[AttributeDef] = []

class RelationTypeDef(BaseModel):
    name:       str
    description: str = ""
    subtypes:   list[SubtypeDef]   = []
    attributes: list[AttributeDef] = []
    head_types: list[str]          # allowed subject entity-type names
    tail_types: list[str]          # allowed object entity-type names

class ExtractionSchema(BaseModel):
    entity_types:   list[EntityTypeDef]
    relation_types: list[RelationTypeDef]
```

That is the complete knob set: entity types (name + description), their subtypes, their per-type
attributes (name + datatype + description); relation types (name + description), subtypes,
per-type attributes, and the entity types allowed as subject (`head_types`) and object
(`tail_types`). Datatypes are limited to `string`/`number`/`boolean`.

The schema is **closed-world**: constrained decoding (below) makes any undeclared type, subtype,
foreign attribute, or wrong datatype literally unrepresentable. Text matching no declared type is
simply not extracted — no `__NOVEL__`/`__UNCLASSIFIED__` escape hatch. Growing the catalogue is a
deliberate edit to the orchestrator's YAML.

---

## Data contracts

### Input — `NerRequest`

```python
class NerRequest(BaseModel):
    text:   str
    schema: ExtractionSchema
```

### Output — `NerResponse`

```python
class ExtractedEntity(BaseModel):
    name:          str           # most complete form in the text
    mention_text:  str           # verbatim substring the model copied (this mention)
    type:          str           # one of the schema's entity types
    subtype:       str | None
    evidence_text: str           # verbatim sentence the mention sits in (context for the resolver)
    attributes:    dict[str, str | float | bool]   # only this type's attributes, correct datatypes
    confidence:    float          # the model's own 0–1 estimate

class ExtractedRelation(BaseModel):
    head:          int           # index into entities[]  (subject)
    tail:          int           # index into entities[]  (object)
    type:          str           # one of the schema's relation types
    subtype:       str | None
    evidence_text: str           # verbatim clause/sentence stating the relation
    attributes:    dict[str, str | float | bool]
    confidence:    float

class NerResponse(BaseModel):
    entities:  list[ExtractedEntity]
    relations: list[ExtractedRelation]
```

No spans, no offsets, no echoed IDs. The model copies `mention_text`/`evidence_text` verbatim;
if the orchestrator wants character positions it locates those strings in its own text. Relation
endpoints are **integer indices** into `entities[]`, not names — the resolver (doc 05) turns them
into names downstream.

---

## Algorithm

### Step 0 (orchestrator, upstream) — pronoun resolution

Extraction quality, especially for relations, drops when a subject/object is a bare pronoun
(*"…Ella transfirió 4,2 M€ a Delta S.A."*). The module provides `coref.resolve_pronouns`, an LLM
rewrite that replaces third-person pronouns with their antecedent's most complete name and
otherwise returns the text **unchanged** (same language, no paraphrase, no added/dropped content).

It runs on the **whole document before chunking** — antecedents are frequently in an earlier
chunk — so the orchestrator calls it upstream:

```
full document → coref.resolve_pronouns() → chunker → /ner (per chunk)
```

`extract` itself never calls coref; the prompt's "do not extract bare pronouns" rule is the
safety net for anything the rewrite leaves behind.

### Step 1 — compile schema → grammar + prompt (`schema_compiler.py`, pure, cached)

Both artefacts derive from `request.schema` and are cached by a hash of it (the same schema is
reused across many calls). The **grammar** is a JSON Schema Ollama turns into a decoding
constraint — a **discriminated union**: one branch per entity type and one per relation type.
Each entity branch pins `type` to a const, lists *only that type's* subtypes (enum) and *only that
type's* attributes (each typed `string→string`, `number→number`, `boolean→boolean`). A leading
`reasoning` string field comes first (a scratchpad, since constrained decoding otherwise forbids
chain-of-thought). This makes three of the four rules impossible to violate.

```jsonc
{
  "type": "object",
  "required": ["reasoning", "entities", "relations"],
  "properties": {
    "reasoning": {"type": "string"},
    "entities":  {"type": "array", "items": {"oneOf": [ /* one branch per entity type */ ]}},
    "relations": {"type": "array", "items": {"oneOf": [ /* one branch per relation type */ ]}}
  }
}
// entity branch:   type=const, subtype ∈ {its subtypes}|null, attributes={its attrs, typed},
//                  name, mention_text, evidence_text, confidence
// relation branch: head:int, tail:int, type=const, subtype, evidence_text, attributes, confidence
```

The **prompt** is the static `extraction.txt` with the schema's type catalogue (type/subtype/
attribute names + descriptions) injected. The compiler builds both; nothing else does.

### Step 2 — call Ollama

```python
raw = ollama_client.extract(system=prompt, user=text, grammar=grammar)
# temperature=0, num_ctx sized to fit prompt + text; reasoning decoded then discarded
```

### Step 3 — check the one runtime rule

The grammar guarantees valid subtypes, attributes, and datatypes. The only thing it cannot
express is which entity type sits at a given relation endpoint index:

| Rule | Action on violation |
|---|---|
| relation `head`/`tail` is a valid index into `entities[]` | drop the relation, log WARNING |
| indexed `head` type ∈ relation's `head_types` and `tail` type ∈ `tail_types` | drop the relation, log WARNING |

Entities are returned as-is. No confidence penalties, no required-attribute logic — structure is
constrained by the grammar, these two checks enforce the one relational rule.

### Step 4 — return `NerResponse`.

---

## Prompt template (`prompts/extraction.txt`)

```
ROLE
====
You are an expert information-extraction system. You read a passage of text and return every
entity and every relation it explicitly states or directly implies, as JSON conforming to the
schema. Never invent information that is not in the text.

Fill the `reasoning` field FIRST: in one or two sentences, note which entities and relations the
text contains. Then fill `entities` and `relations`. Do all thinking in `reasoning` — you cannot
write anything outside the JSON.

TYPES
=====
{type_catalogue}
(Only the types listed here exist. If something matches no type, do not extract it.)

WHAT TO EXTRACT
===============
- Entities: every distinct real-world entity that matches a type above — named mentions and
  definite descriptions that identify a specific entity ("the finance minister"). Do NOT extract
  bare pronouns ("he", "it") or generic nouns ("a company", "investors", "money").
- Relations: only those whose subject AND object are entities you listed, and which the text
  states or directly implies. Never infer relations from world knowledge.

FIELDS
======
- name: the most complete form of the entity in the text. Preserve original language/spelling.
- mention_text: copy the exact surface string for this mention, character-for-character.
- head / tail: the INTEGER POSITIONS (0-based) of the subject and object in the `entities` array
  you are producing — not names.
- evidence_text: copy verbatim — for an entity, the sentence the mention appears in; for a
  relation, the clause that states it.
- subtype: use only when the text clearly supports it; otherwise null.
- attributes: fill one only when its value is explicitly stated; never guess; leave unstated
  attributes out. Normalise dates/codes only when the conversion is certain.
- confidence: your own 0–1 estimate. 0.9–1.0 explicit, 0.6–0.8 implied, <0.3 don't extract.
```

---

## Ollama client (`nlp/ner/ollama_client.py`)

Separate from the summarizer's client so models/timeouts are independent.

```python
import os, json, httpx

OLLAMA_HOST       = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
EXTRACTION_MODEL  = os.environ.get("EXTRACTION_MODEL",   "qwen2.5:32b")
EXTRACTION_TIMEOUT= float(os.environ.get("EXTRACTION_TIMEOUT", "180"))
EXTRACTION_NUM_CTX= int(os.environ.get("EXTRACTION_NUM_CTX", "8192"))


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

Retry up to 3 attempts, backoff `2 ** attempt`, on `httpx.HTTPError` / `json.JSONDecodeError` /
`KeyError`. `num_ctx` must hold prompt + text or Ollama **silently truncates** — size it to the
schema (8192 is ample for a handful of types; raise for large schemas). The coref client reuses
the same host/model.

---

## API (`api/routers/ner.py`)

```python
from fastapi import APIRouter, HTTPException
import httpx

from api.models import NerRequest, NerResponse
from api.warmth import mark_warm
from nlp.ner import service as ner_service

router = APIRouter()


@router.post("/ner", response_model=NerResponse)
def ner(req: NerRequest) -> NerResponse:
    if not req.text.strip():
        raise HTTPException(422, "text must be non-empty")
    try:
        result = ner_service.run(req.text, req.schema)
    except httpx.HTTPError:
        raise HTTPException(503, "ollama_unavailable")
    mark_warm("ner")
    return result
```

Register in `api/main.py` (`from api.routers import ner as ner_router` / `app.include_router(
ner_router.router)`). In-process orchestrators may call `ner_service.run(text, schema)` directly —
no hidden state either way.

`NerRequest`/`NerResponse`/`ExtractedEntity`/`ExtractedRelation` and the `ExtractionSchema` types
are added to `api/models.py` (or imported from `nlp/ner/schema_types.py`).

---

## Configuration

| Env var | Default | Description |
|---|---|---|
| `EXTRACTION_MODEL` | `qwen2.5:32b` | Ollama model for extraction + coref |
| `EXTRACTION_TIMEOUT` | `180` | Seconds per Ollama call |
| `EXTRACTION_NUM_CTX` | `8192` | Context window; must hold prompt + text |
| `OLLAMA_HOST` | `http://ollama:11434` | Shared with summarizer |

---

## Dependencies

- Ollama sidecar — same container the summarizer uses.
- `httpx` — already in requirements.
- No config loader, no schema YAML on the extraction path, no new Python packages.
