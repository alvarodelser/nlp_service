# Summarizer Module — Expansion for Entity Description Maintenance

## Purpose of this document

The existing `nlp/summarizer/` module rewrites article headlines and summaries using
Ollama. This document covers the **additive expansion** for entity description maintenance
— the GraphRAG-style step that refreshes a canonical entity's description as new evidence
accumulates across documents. (It is the *only* module that writes prose descriptions; the
extractor and resolver just copy verbatim evidence text. Edges have no description — they
are identified by their type, endpoints, and verbatim evidence.)

The existing article summarization code is **not changed**. The only modification is a
one-line default-argument change to `ollama_client.generate()` that is fully backward-
compatible with all existing callers.

---

## What changes and what does not

| Component | Status |
|---|---|
| `ollama_client.py` — `generate()` signature | One new optional parameter `schema` with default = existing article schema. No existing caller changes. |
| `service.py` — `run()` | Unchanged |
| `service.py` — `describe_entity()` | **New function** appended |
| `prompts/rewrite.es.txt` | Unchanged |
| `prompts/entity_describe.txt` | **New prompt template** |
| `validator.py` | Unchanged |
| `api/routers/summarize.py` — `POST /summarize` | Unchanged |
| `api/routers/summarize.py` — `POST /summarize/entity` | **New endpoint** appended |
| `api/models.py` | **New models** appended; existing models unchanged |

---

## Why entity description needs its own summarizer step

When a canonical entity accrues mentions across many documents, its description in the
graph is initialised from the first document's clustering output. As more evidence arrives,
that initial description becomes stale or incomplete. This step re-summarises the entity's
description from all accumulated context, keeping the graph node useful for retrieval
(GraphRAG-style).

This is triggered by the persistence module after a successful merge into an existing node,
not on every extraction. The NLP service is stateless with respect to triggering — it
receives a payload and returns an updated description.

---

## Changes to `ollama_client.py`

Current signature:
```python
def generate(prompt: str, max_retries: int = 3, timeout: float = _TIMEOUT) -> dict:
```

New signature (one added parameter, default preserves existing behaviour):
```python
_ARTICLE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "summary":  {"type": "string"},
    },
    "required": ["headline", "summary"],
}

def generate(
    prompt:      str,
    schema:      dict       = _ARTICLE_SCHEMA,
    max_retries: int        = 3,
    timeout:     float      = _TIMEOUT,
) -> dict:
```

The body does not change — `schema` is passed as `"format": schema` in the Ollama request
body, exactly as before. The rename from the inline literal to `_ARTICLE_SCHEMA` is the
only other change, and it does not affect runtime behaviour.

Existing caller in `service.py`:
```python
result = ollama_client.generate(prompt)   # still works; gets _ARTICLE_SCHEMA by default
```

---

## New prompt template (`prompts/entity_describe.txt`)

```
You are maintaining a knowledge graph for investigative journalism.
Write a concise, factual description of the entity below based on the evidence provided.

ENTITY
======
Name: {name}
Type: {entity_type}
{subtype_line}

ACCUMULATED EVIDENCE
====================
{evidence_list}

RULES
=====
- Write 2–4 sentences maximum.
- State only what the evidence explicitly supports.
- Include key relationships, roles, and any confirmed financial or legal facts.
- Do not speculate or add information not present in the evidence.
- Write the description in Spanish, regardless of the evidence language.
```

`{evidence_list}` is formatted as a numbered list of context strings, one per document
mention:

```
1. [doc_id: abc123, 2024-03-15] "Acme Holdings, a Delaware-registered shell company,
   received €4.2M from the minister's personal account..."
2. [doc_id: def456, 2024-04-01] "The fund, Acme Holdings Ltd, was dissolved in May 2024..."
```

`{subtype_line}` is `Subtype: {subtype}` if present, otherwise omitted.

---

## New schema for entity descriptions

```python
_ENTITY_DESCRIBE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
    },
    "required": ["description"],
}
```

---

## New function `service.describe_entity()`

Append to the bottom of `nlp/summarizer/service.py`. The existing `run()` function is
not touched.

```python
_ENTITY_DESCRIBE_PROMPT_PATH = Path(__file__).parent / "prompts" / "entity_describe.txt"
_ENTITY_DESCRIBE_TEMPLATE: str | None = None

def _load_entity_describe_template() -> str:
    global _ENTITY_DESCRIBE_TEMPLATE
    if _ENTITY_DESCRIBE_TEMPLATE is None:
        _ENTITY_DESCRIBE_TEMPLATE = _ENTITY_DESCRIBE_PROMPT_PATH.read_text(encoding="utf-8")
    return _ENTITY_DESCRIBE_TEMPLATE


def describe_entity(
    name:        str,
    entity_type: str,
    subtype:     str | None,
    evidence:    list[str],   # pre-formatted evidence strings (caller's responsibility)
) -> dict:
    """Returns {'description': str}.

    Raises httpx.HTTPError if Ollama is unavailable.
    Raises ValueError if the LLM returns malformed JSON (after retries).
    """
    template = _load_entity_describe_template()
    subtype_line = f"Subtype: {subtype}" if subtype else ""
    evidence_list = "\n".join(f"{i+1}. {e}" for i, e in enumerate(evidence))
    prompt = template.format(
        name=name,
        entity_type=entity_type,
        subtype_line=subtype_line,
        evidence_list=evidence_list,
    )
    result = ollama_client.generate(prompt, schema=_ENTITY_DESCRIBE_SCHEMA)
    return {"description": result["description"]}
```

No validator step is applied to entity descriptions (the length constraints from article
summarization do not apply here). If the LLM returns an invalid structure, `generate()`
raises `ValueError` after retries — the caller handles it.

---

## New API endpoint (`api/routers/summarize.py`)

Append after the existing `/summarize` route. Existing route is not modified.

```python
from api.models import EntityDescribeRequest, EntityDescribeResponse

@router.post("/summarize/entity", response_model=EntityDescribeResponse)
def summarize_entity(req: EntityDescribeRequest) -> EntityDescribeResponse:
    if not req.evidence:
        raise HTTPException(422, "evidence must be non-empty")
    try:
        result = summarizer_service.describe_entity(
            name=req.name,
            entity_type=req.entity_type,
            subtype=req.subtype,
            evidence=req.evidence,
        )
    except httpx.HTTPError as exc:
        log.error("ollama unavailable: %s", exc)
        raise HTTPException(503, "ollama_unavailable")
    except (KeyError, ValueError) as exc:
        log.error("ollama json format failed: %s", exc)
        raise HTTPException(503, "ollama_json_format_failed")
    return EntityDescribeResponse(
        canonical_id=req.canonical_id,
        description=result["description"],
    )
```

---

## New Pydantic models (`api/models.py`)

Append after existing models:

```python
# --- Entity Description ---

class EntityDescribeRequest(BaseModel):
    canonical_id: str             # graph node being updated
    name:         str             # canonical name
    entity_type:  str
    subtype:      str | None = None
    evidence:     list[str]       # pre-formatted context strings, 1–50 items

class EntityDescribeResponse(BaseModel):
    canonical_id: str
    description:  str
```

Evidence formatting is the caller's responsibility. A recommended format:

```
[doc_id: {doc_id}, {date}] "{excerpt from chunk surrounding the entity mention}"
```

Keeping formatting outside the NLP service makes it easier to change evidence representation
without updating the service.

---

## Triggering

`POST /summarize/entity` is triggered by the persistence module (out of scope) in two situations:

1. A `"merge"` decision from the disambiguator causes a new document's mention to be
   merged into an existing canonical node → re-describe with all accumulated evidence.
2. A new canonical node is created from a `ResolvedEntity` with multiple supporting mentions →
   generate an initial consolidated description.

The NLP service has no knowledge of when to trigger this — it only responds to calls.

---

## Configuration

No new environment variables. The endpoint uses the same Ollama model as article
summarization (`OLLAMA_MODEL`) and the same timeout (`OLLAMA_TIMEOUT`). If entity
description needs a different model, add `ENTITY_DESCRIBE_MODEL` and update
`describe_entity()` to pass it through `generate()`.

---

## Testing

See [06-testing.md](06-testing.md) §5 — mock Ollama via `httpx.post`; verify that
`describe_entity()` passes the entity schema (not the article schema) to `generate()`.

---

## Dependencies

- No new packages
- No new models loaded into process memory
- Same Ollama sidecar as existing summarizer
