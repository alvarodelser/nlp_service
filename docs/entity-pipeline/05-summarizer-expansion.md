# Summarizer Module — Unified Endpoint

## Purpose of this document

Refactors the existing `nlp/summarizer/` module to expose a **single unified
`POST /summarize` endpoint** parameterised by `type`. Article rewriting, entity
description generation, and any future summary type (relations, edges, clusters) all
route through the same endpoint. Prompts live in the service.

The existing article summarisation logic is **not changed in behaviour**. Only the
API surface and service dispatch are unified.

---

## What changes

| Component | Status |
|---|---|
| `ollama_client.py` — `generate()` | One new optional `schema` parameter (default = article schema). No existing caller changes. |
| `service.py` — `run()` | Unchanged |
| `service.py` — `describe_entity()` | **New function** appended |
| `prompts/rewrite.es.txt` | Unchanged |
| `prompts/entity_describe.txt` | **New** |
| `validator.py` | Unchanged |
| `api/routers/summarize.py` — `POST /summarize` | **Unified** — replaces both old `/summarize` and new `/summarize/entity` |
| `api/models.py` | **Unified** request/response models (discriminated union) |

---

## Unified API

### `POST /summarize`

```python
# --- Requests ---

class ArticleSummarizeRequest(BaseModel):
    type:    Literal["article"] = "article"
    text:    str           # article body to rewrite
    lang:    str = "es"   # target language for rewrite prompt

class EntityDescribeRequest(BaseModel):
    type:         Literal["entity"] = "entity"
    canonical_id: str             # graph node being updated
    name:         str
    entity_type:  str
    subtype:      str | None = None
    evidence:     list[str]       # pre-formatted context strings, 1–50 items

SummarizeRequest = Annotated[
    ArticleSummarizeRequest | EntityDescribeRequest,
    Field(discriminator="type")
]

# --- Responses ---

class ArticleSummarizeResponse(BaseModel):
    type:     Literal["article"] = "article"
    headline: str
    summary:  str

class EntityDescribeResponse(BaseModel):
    type:         Literal["entity"] = "entity"
    canonical_id: str
    description:  str

SummarizeResponse = Annotated[
    ArticleSummarizeResponse | EntityDescribeResponse,
    Field(discriminator="type")
]
```

Router:

```python
@router.post("/summarize")
def summarize(req: SummarizeRequest) -> SummarizeResponse:
    if req.type == "article":
        if not req.text.strip():
            raise HTTPException(422, "text must be non-empty")
        result = summarizer_service.run(req.text, req.lang)
        return ArticleSummarizeResponse(**result)

    if req.type == "entity":
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
        return EntityDescribeResponse(canonical_id=req.canonical_id, **result)
```

The old `POST /summarize/entity` is removed. Callers migrate to `POST /summarize` with
`"type": "entity"`.

---

## Changes to `ollama_client.py`

Add one optional parameter with a default that preserves the current article schema.
Existing callers need no changes.

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
    schema:      dict  = _ARTICLE_SCHEMA,
    max_retries: int   = 3,
    timeout:     float = _TIMEOUT,
) -> dict:
    ...   # body unchanged; schema passed as "format": schema in Ollama request
```

---

## New function `service.describe_entity()`

Appended to the bottom of `nlp/summarizer/service.py`. The existing `run()` is untouched.

```python
_ENTITY_DESCRIBE_SCHEMA: dict = {
    "type": "object",
    "properties": {"description": {"type": "string"}},
    "required": ["description"],
}

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
    evidence:    list[str],
) -> dict:
    """Returns {'description': str}."""
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
- Write in English regardless of the evidence language.
```

Evidence items are pre-formatted by the caller:

```
[doc_id: abc123, 2024-03-15] "Acme Holdings, a Delaware-registered shell company,
   received €4.2M from the minister's personal account..."
```

---

## Triggering

`POST /summarize` with `type: "entity"` is called by the persistence module (out of scope)
in two situations:

1. A `"merge"` decision from the disambiguator → re-describe with all accumulated evidence.
2. A new canonical node is created → generate its initial description.

The NLP service has no knowledge of when to trigger — it only responds to calls.

---

## Extending to new types

To add a `"relation"` or `"edge"` summary type:

1. Add a new `RelationDescribeRequest` / `RelationDescribeResponse` Pydantic model.
2. Extend the `SummarizeRequest` / `SummarizeResponse` union.
3. Add a new prompt template under `prompts/`.
4. Add a new service function analogous to `describe_entity()`.
5. Add a branch in the router.

No existing code paths change.

---

## Configuration

No new environment variables. Uses the same `OLLAMA_MODEL` and `OLLAMA_TIMEOUT` as
article summarisation. Add `ENTITY_DESCRIBE_MODEL` to override for entity descriptions
only if needed.

---

## Testing

See [06-testing.md](06-testing.md) §5 — mock Ollama via `httpx.post`. Verify that
`describe_entity()` passes `_ENTITY_DESCRIBE_SCHEMA` (not `_ARTICLE_SCHEMA`) to
`generate()`. Verify router dispatches correctly on `type` discriminator.

---

## Dependencies

- No new packages
- Same Ollama sidecar as existing summariser
