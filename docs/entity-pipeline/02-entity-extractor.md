# Entity Extractor — Design & Implementation

## Purpose

Pass 1 of the document-level extraction pipeline. Takes a single chunk (produced by the
chunker with doc provenance attached) and returns all entity mentions and relation instances
found in that chunk, typed against the ontology and structured as guaranteed-valid JSON.

This is the only module that calls the LLM per-chunk. Everything else in the pipeline
operates on the structured output of this module.

---

## Position in pipeline

```
Chunker  →  [Entity Extractor]  →  Normalizer  →  Disambiguator  →  Persistence
             (per chunk, Pass 1)    (per doc, Pass 2)
```

---

## File layout

```
nlp/
  entity_extractor/
    __init__.py
    service.py          ← orchestration: build prompt, call Ollama, post-validate, return
    prompts/
      extraction.txt    ← system prompt template (injected with ontology descriptions)
    ollama_client.py    ← thin wrapper around /api/chat with format= JSON Schema
                           (separate from summarizer's client to allow independent config)

api/
  routers/
    entity_extract.py   ← POST /entity-extract
```

---

## Data contracts

### Request — `POST /entity-extract`

```python
class EntityExtractRequest(BaseModel):
    doc_id:      str           # sha256 of the source document
    chunk_id:    str           # doc_id + "_" + zero-padded chunk index
    text:        str           # chunk text (typically 300–600 tokens)
    char_start:  int           # absolute character offset in original doc
    char_end:    int           # absolute character offset in original doc
    use_case:    str           # ontology identifier, e.g. "financial_flows"
    source_meta: dict = {}     # passed through for provenance, not used in extraction
```

### Response — `EntityExtractResponse`

```python
class ExtractedSpan(BaseModel):
    start: int    # character offset relative to chunk start
    end:   int

class ExtractedEntity(BaseModel):
    name:        str
    type:        str            # validated against ontology enum
    subtype:     str | None
    description: str
    span:        ExtractedSpan
    confidence:  float

class RelationAttributes(BaseModel):
    amount:    float | None = None
    currency:  str   | None = None
    date:      str   | None = None
    direction: str   | None = None
    # additional keys allowed for other use cases

class ExtractedRelation(BaseModel):
    head:          str             # entity name as written in the chunk
    relation:      str             # canonical type, validated against ontology enum
    tail:          str
    description:   str
    evidence_span: ExtractedSpan
    attributes:    RelationAttributes = RelationAttributes()
    confidence:    float

class EntityExtractResponse(BaseModel):
    doc_id:    str
    chunk_id:  str
    char_start: int
    char_end:   int
    entities:  list[ExtractedEntity]
    relations: list[ExtractedRelation]
```

Spans in the response are **relative to the chunk**. The normalizer or any downstream
consumer adds `char_start` to convert to absolute document offsets. This keeps the
extraction contract simple and the offset arithmetic in one place.

---

## Algorithm

### 1. Build system prompt

```
extraction.txt template injected with:
  - ontology.system_prompt()   → all entity/relation type descriptions + subtype rules
  - instructions for confidence scoring
  - instruction to set span relative to chunk start (character 0)
  - instruction to use null for optional attributes when not stated
```

The system prompt is built once per `(use_case, ontology_version)` and cached.

### 2. User message

```
CHUNK [{chunk_id}] ({char_start}–{char_end}):

{text}
```

### 3. Ollama call

```python
ollama_client.extract(
    system=system_prompt,
    user=user_message,
    schema=onto.extraction_schema(),   # JSON Schema enforces types as enums
    model=EXTRACTION_MODEL,
    timeout=EXTRACTION_TIMEOUT,
)
```

Uses `/api/chat` (not `/api/generate`) so system and user roles are distinct. The `format=`
parameter receives the full JSON Schema; Ollama's GBNF grammar enforces structural validity.

### 4. Post-extraction validation

Structural validity is guaranteed by constrained decoding. Semantic checks are applied
before returning:

| Check | Action on failure |
|---|---|
| `span.end > span.start` | set `confidence = 0.0`, log WARNING |
| `span.end <= len(text)` | clamp to `len(text)`, log WARNING |
| `subtype` belongs to parent `type` | set `subtype = null`, log WARNING |
| relation `head` or `tail` not found in `entities[].name` | log WARNING, keep relation |
| `relation` `head_types`/`tail_types` violated | log WARNING, keep relation |

No extraction is silently dropped. Failures are flagged in logs and via `confidence=0.0`
so the normalizer can filter them.

### 5. Return

Return `EntityExtractResponse`. The router logs `doc_id`, `chunk_id`, entity count,
relation count, and Ollama latency.

---

## Prompt template (`extraction.txt`)

```
You are an investigative data analyst extracting structured information from a document chunk
for a knowledge graph. Extract only information explicitly stated or strongly implied in the
chunk. Do not hallucinate.

ONTOLOGY
========
{ontology_descriptions}

RULES
=====
- Assign confidence between 0.0 (very uncertain) and 1.0 (certain).
- Use subtype only when you are confident; set to null otherwise.
- For relations, capture every attribute that is explicitly stated; set others to null.
- Spans are CHARACTER OFFSETS relative to the start of the chunk (first character = 0).
- A relation's head and tail must be entity names you have already listed.
- If a relation's type does not fit any ontology type, do not extract it.
- Extract all entity mentions, including pronouns and aliases — the resolver will cluster them.
```

---

## Ollama client (`entity_extractor/ollama_client.py`)

Separate from `nlp/summarizer/ollama_client.py` so extraction and summarization can use
different models and timeouts independently.

```python
EXTRACTION_MODEL   = os.environ.get("EXTRACTION_MODEL",   "qwen2.5:32b")
EXTRACTION_TIMEOUT = float(os.environ.get("EXTRACTION_TIMEOUT", "180"))

def extract(system: str, user: str, schema: dict, ...) -> dict:
    response = httpx.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model":    EXTRACTION_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "stream": False,
            "format": schema,
        },
        timeout=EXTRACTION_TIMEOUT,
    )
    response.raise_for_status()
    return json.loads(response.json()["message"]["content"])
```

Retry logic: up to 3 attempts with exponential backoff (1s, 2s) on `httpx.HTTPError`.
`json.JSONDecodeError` after a successful HTTP response is a constrained-decoding failure —
log at ERROR, raise (the router returns 503).

---

## API router (`api/routers/entity_extract.py`)

```python
@router.post("/entity-extract", response_model=EntityExtractResponse)
def entity_extract(req: EntityExtractRequest) -> EntityExtractResponse:
    if not req.text.strip():
        raise HTTPException(422, "text must be non-empty")
    onto = ontology.load(req.use_case)
    try:
        result = entity_extractor_service.run(req, onto)
    except httpx.HTTPError as exc:
        raise HTTPException(503, "ollama_unavailable")
    return result
```

---

## Configuration

| Env var | Default | Description |
|---|---|---|
| `EXTRACTION_MODEL` | `qwen2.5:32b` | Ollama model tag for extraction |
| `EXTRACTION_TIMEOUT` | `180` | Seconds per Ollama call |
| `OLLAMA_HOST` | `http://ollama:11434` | Shared with summarizer |

---

## Dependencies

- `nlp/ontology.py` — for schema and system prompt
- Ollama sidecar — same container already used by summarizer
- `httpx` — already in requirements.txt
- No new Python packages
