# Summarizer Module — Unified Endpoint

## Purpose

Refactors `nlp/summarizer/` so that article rewriting, entity description generation,
and relation description generation all go through a single `POST /summarize` endpoint
parameterised by `type`. Prompts live in the service; the caller only supplies the content
and evidence.

The existing article rewriting logic is not changed in behaviour. Only the API surface and
service dispatch are unified.

---

## The GraphRAG evidence pattern

Entity and relation descriptions in the knowledge graph are not static. Each time a
canonical entity is updated (a new document merges a mention into it), the description
should be refreshed to incorporate the new evidence. This is the GraphRAG pattern:
accumulate text windows, regenerate the description from all of them.

**Text window** — the sentence(s) surrounding an entity or relation mention in a chunk.
The clusterer already has the raw material:

```
LocalEntity.mentions[i].char_start / char_end   ← span inside the chunk
ChunkExtraction.text                             ← full chunk text
```

Extracting ±2 sentences around the span from the chunk text gives the evidence window.
For relations, `LocalEdge.evidence.text` already is the extracted evidence sentence.

**Storage (persistence module's responsibility, not the NLP service):**
The Weaviate entity and edge nodes each carry an `evidence_windows: text[]` property
(see `01-ontology.md` — Weaviate collection spec). After a successful merge or create,
the persistence module appends the new window(s) to that property. Before calling
`POST /summarize`, the caller reads `evidence_windows` from Weaviate and passes them
as the `evidence` list.

**Recommended evidence string format:**

```
[doc_id: abc123, 2024-03-15] "Acme Holdings, a Delaware-registered shell company,
received €4.2M from the minister's personal account via a Maltese correspondent bank."
```

The service does not enforce format — it is injected verbatim into the prompt. Using
doc_id and date helps the LLM situate the evidence in time but is optional.

---

## What changes

| Component | Status |
|---|---|
| `ollama_client.py` — `generate()` | One new optional `schema` parameter (default = article schema). No existing caller changes. |
| `service.py` — `run()` | Unchanged |
| `service.py` — `describe_entity()` | **New** |
| `service.py` — `describe_relation()` | **New** |
| `prompts/rewrite.es.txt` | Unchanged |
| `prompts/entity_describe.txt` | **New** |
| `prompts/relation_describe.txt` | **New** |
| `validator.py` | Unchanged |
| `api/routers/summarize.py` — `POST /summarize` | **Unified** — replaces both old `/summarize` and the separate `/summarize/entity` |
| `api/models.py` | **Unified** discriminated-union request/response |

---

## Unified API — `POST /summarize`

### Request models

```python
class ArticleSummarizeRequest(BaseModel):
    type: Literal["article"] = "article"
    text: str           # article body to rewrite
    lang: str = "es"    # target language for rewrite prompt

class EntityDescribeRequest(BaseModel):
    type:         Literal["entity"] = "entity"
    canonical_id: str             # graph node being updated (echoed in response)
    name:         str
    entity_type:  str
    subtype:      str | None = None
    evidence:     list[str]       # pre-formatted text windows, 1–50 items

class RelationDescribeRequest(BaseModel):
    type:         Literal["relation"] = "relation"
    canonical_id: str             # canonical edge ID (echoed in response)
    head_name:    str
    head_type:    str
    relation:     str             # relation type name from schema (e.g. "PAYMENT_TO")
    tail_name:    str
    tail_type:    str
    evidence:     list[str]       # pre-formatted text windows, 1–50 items

SummarizeRequest = Annotated[
    ArticleSummarizeRequest | EntityDescribeRequest | RelationDescribeRequest,
    Field(discriminator="type")
]
```

### Response models

```python
class ArticleSummarizeResponse(BaseModel):
    type:     Literal["article"] = "article"
    headline: str
    summary:  str

class EntityDescribeResponse(BaseModel):
    type:         Literal["entity"] = "entity"
    canonical_id: str
    description:  str

class RelationDescribeResponse(BaseModel):
    type:         Literal["relation"] = "relation"
    canonical_id: str
    description:  str

SummarizeResponse = Annotated[
    ArticleSummarizeResponse | EntityDescribeResponse | RelationDescribeResponse,
    Field(discriminator="type")
]
```

### Router

```python
@router.post("/summarize")
def summarize(req: SummarizeRequest) -> SummarizeResponse:
    if req.type == "article":
        if not req.text.strip():
            raise HTTPException(422, "text must be non-empty")
        result = summarizer_service.run(req.text, req.lang)
        return ArticleSummarizeResponse(**result)

    if req.type in ("entity", "relation"):
        if not req.evidence:
            raise HTTPException(422, "evidence must be non-empty")
        try:
            if req.type == "entity":
                result = summarizer_service.describe_entity(req)
                return EntityDescribeResponse(canonical_id=req.canonical_id, **result)
            else:
                result = summarizer_service.describe_relation(req)
                return RelationDescribeResponse(canonical_id=req.canonical_id, **result)
        except httpx.HTTPError as exc:
            log.error("ollama unavailable: %s", exc)
            raise HTTPException(503, "ollama_unavailable")
        except (KeyError, ValueError) as exc:
            log.error("ollama json format failed: %s", exc)
            raise HTTPException(503, "ollama_json_format_failed")
```

---

## Change to `ollama_client.py`

One new optional parameter, default preserves current article schema so no existing
caller changes:

```python
_ARTICLE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "summary":  {"type": "string"},
    },
    "required": ["headline", "summary"],
}

_DESCRIBE_SCHEMA: dict = {
    "type": "object",
    "properties": {"description": {"type": "string"}},
    "required": ["description"],
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

## New service functions

Both are appended to the bottom of `service.py`. Existing `run()` is not touched.

### `describe_entity(req: EntityDescribeRequest) -> dict`

Builds a prompt from the `entity_describe.txt` template and calls
`ollama_client.generate(prompt, schema=_DESCRIBE_SCHEMA)`.

### `describe_relation(req: RelationDescribeRequest) -> dict`

Builds a prompt from the `relation_describe.txt` template and calls
`ollama_client.generate(prompt, schema=_DESCRIBE_SCHEMA)`.

Neither function applies the article validator (length constraints do not apply to
graph descriptions). If the LLM returns malformed JSON after retries, `generate()`
raises `ValueError` — the router returns 503.

---

## Prompt templates

### `prompts/entity_describe.txt`

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
- Include key roles, relationships, and any confirmed financial or legal facts.
- Do not speculate or add information not in the evidence.
- Write in English regardless of the source language.
```

`{subtype_line}` is `Subtype: {subtype}` if present, otherwise omitted.
`{evidence_list}` is a numbered list, one item per evidence string.

### `prompts/relation_describe.txt`

```
You are maintaining a knowledge graph for investigative journalism.
Write a concise, factual description of the relationship below based on the evidence provided.

RELATIONSHIP
============
From: {head_name} ({head_type})
Relation: {relation}
To:   {tail_name} ({tail_type})

ACCUMULATED EVIDENCE
====================
{evidence_list}

RULES
=====
- Write 2–4 sentences maximum.
- State only what the evidence explicitly supports.
- Include amounts, dates, currencies, and mechanisms (wire, cash, crypto) where stated.
- Do not speculate or add information not in the evidence.
- Write in English regardless of the source language.
```

---

## When each type is triggered

| Type | Triggered by |
|---|---|
| `"article"` | Ingestion pipeline after chunking |
| `"entity"` | Persistence module after a merge into an existing canonical node, or on creation of a new node with multiple supporting chunks |
| `"relation"` | Persistence module after a canonical edge receives a new evidence window (same trigger as entity, but on the edge) |

The NLP service has no knowledge of when to trigger — it only responds to calls.

---

## Extending to new types

To add a future type (e.g. `"cluster_summary"` or `"timeline"`):

1. Add a new `XxxRequest` / `XxxResponse` Pydantic model with `type: Literal["xxx"]`.
2. Extend the `SummarizeRequest` / `SummarizeResponse` union.
3. Add a new prompt template under `prompts/`.
4. Add a new `describe_xxx()` service function.
5. Add one branch in the router.

Nothing else changes.

---

## Configuration

No new environment variables. Uses the same `OLLAMA_MODEL` and `OLLAMA_TIMEOUT` as
article summarisation.

---

## Testing

See [06-testing.md](06-testing.md) §5. Mock Ollama via `httpx.post`. Key cases:
- `describe_entity()` passes `_DESCRIBE_SCHEMA`, not `_ARTICLE_SCHEMA`
- `describe_relation()` passes `_DESCRIBE_SCHEMA`
- Router dispatches correctly on `type` discriminator
- Empty `evidence` → 422
- Ollama unavailable → 503 on entity and relation paths

---

## Dependencies

- No new packages
- Same Ollama sidecar as existing summariser
