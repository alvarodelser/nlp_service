# Summarizer — Implementation Doc

**Module:** `nlp/summarizer/` · **Status:** Reworked (supersedes the old summarizer-expansion doc)
**Authoritative source:** `docs/superpowers/specs/2026-06-05-nlp-two-pipelines-design.md` §3.1

## Purpose

The summarizer is the **only module that writes prose**. It is a single generation primitive
driven by **stored named profiles**: the caller picks a profile and passes that profile's
fields; the module owns the prompt template, the output JSON schema, the validation rules, and
the extractive pre-step. It serves four tasks today:

| Profile | In | Out |
|---|---|---|
| `article` | `title`, `text` | `headline`, `summary` |
| `entity_desc` | `name`, `type`, `subtype?`, `evidence[]` | `description` (Spanish) |
| `relation_desc` | `head`, `tail`, `type`, `subtype?`, `attributes`, `evidence[]` | `description` (Spanish) |
| `aggregate` | `name`, `type`, `descriptions[]`, `evidence[]` | `description` (Spanish) |

Two behavioural changes versus the old code drive this rework:

1. **The extractive step is baked in.** The old `/extract` service (TextRank, `nlp/extractor/`)
   is **retired**; its `extractive.py` moves into the summarizer and runs **internally, only when
   the profile's long input field exceeds a word budget**. Callers never pass a precomputed
   `extract` anymore.
2. **Embedding moves out.** `/summarize` no longer calls `encode()`. Embeddings are the
   `vectorizer` service's job (bge-m3), invoked separately by the orchestrator.

---

## What changes and what does not

| Component | Status |
|---|---|
| `nlp/summarizer/service.py` — `run()` | **Replaced** by `summarize(profile, fields)` |
| `nlp/summarizer/profiles.py` | **New** — profile registry |
| `nlp/summarizer/extractive.py` | **New** — moved verbatim from `nlp/extractor/extractive.py` |
| `nlp/summarizer/ollama_client.py` — `generate()` | **Signature change**: add `schema` param |
| `nlp/summarizer/validator.py` | Unchanged (used by the `article` profile only) |
| `nlp/summarizer/prompts/rewrite.es.txt` | **Renamed** → `prompts/article.es.txt` |
| `nlp/summarizer/prompts/entity_desc.es.txt` | **New** |
| `nlp/summarizer/prompts/relation_desc.es.txt` | **New** |
| `nlp/summarizer/prompts/aggregate.es.txt` | **New** |
| `api/routers/summarize.py` | **Replaced**: profile + fields in, no embedding |
| `api/models.py` — `SummarizeRequest/Response` | **Replaced** |
| `api/models.py` — `ExtractRequest/Response` | **Deleted** |
| `api/routers/extract.py` | **Deleted** |
| `nlp/extractor/` (whole module) | **Deleted** (after `extractive.py` is moved) |

> Before deleting `nlp/extractor/`, confirm nothing else imports it:
> `grep -rn "nlp.extractor" --include=*.py .` should return only `api/routers/extract.py`
> and `nlp/extractor/` itself. The router that registers `/extract` in `api/main.py` must also
> be removed.

---

## File layout (after rework)

```
nlp/
  summarizer/
    __init__.py
    service.py          ← summarize(profile, fields): dispatch → extractive pre-step → LLM → validate
    profiles.py         ← NEW: the Profile dataclass + PROFILES registry
    extractive.py       ← NEW: moved from nlp/extractor/extractive.py (sumy TextRank), unchanged
    ollama_client.py    ← generate(prompt, schema=...) — schema now a parameter
    validator.py        ← unchanged; article length validation
    prompts/
      article.es.txt        ← renamed from rewrite.es.txt
      entity_desc.es.txt    ← NEW
      relation_desc.es.txt  ← NEW
      aggregate.es.txt      ← NEW
api/
  routers/
    summarize.py        ← POST /summarize {profile, fields} → {result}
```

---

## Profile registry (`nlp/summarizer/profiles.py`)

A profile is immutable config: its prompt path, output schema, the keys to return, an optional
validator + retry suffix, and the extractive budget (which input field is reduced, and at what
word count). All domain knowledge for a task lives here — `service.py` stays generic.

```python
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import validator

_PROMPT_DIR = Path(__file__).parent / "prompts"


@dataclass(frozen=True)
class Profile:
    name:          str
    prompt_file:   str                         # under prompts/
    schema:        dict                        # Ollama output JSON schema
    output_keys:   tuple[str, ...]             # keys lifted from the LLM result
    extract_field: str | None                  # long input field reduced when over the budget
    validate:      Callable[[dict], tuple[bool, str | None]] | None = None
    retry_suffix:  str = ""


_ARTICLE_SCHEMA = {
    "type": "object",
    "properties": {"headline": {"type": "string"}, "summary": {"type": "string"}},
    "required": ["headline", "summary"],
}
_DESCRIPTION_SCHEMA = {
    "type": "object",
    "properties": {"description": {"type": "string"}},
    "required": ["description"],
}

PROFILES: dict[str, Profile] = {
    "article": Profile(
        name="article",
        prompt_file="article.es.txt",
        schema=_ARTICLE_SCHEMA,
        output_keys=("headline", "summary"),
        extract_field="text",
        validate=validator.validate,
        retry_suffix="\nRECUERDA: titular 8-15 palabras, resumen 2-4 frases.",
    ),
    "entity_desc": Profile(
        name="entity_desc",
        prompt_file="entity_desc.es.txt",
        schema=_DESCRIPTION_SCHEMA,
        output_keys=("description",),
        extract_field="evidence_text",   # the joined evidence list (see service.py)
    ),
    "relation_desc": Profile(
        name="relation_desc",
        prompt_file="relation_desc.es.txt",
        schema=_DESCRIPTION_SCHEMA,
        output_keys=("description",),
        extract_field="evidence_text",
    ),
    "aggregate": Profile(
        name="aggregate",
        prompt_file="aggregate.es.txt",
        schema=_DESCRIPTION_SCHEMA,
        output_keys=("description",),
        extract_field="evidence_text",
    ),
}


def get(profile_name: str) -> Profile:
    try:
        return PROFILES[profile_name]
    except KeyError:
        raise ValueError(f"unknown summarizer profile: {profile_name!r}")
```

---

## Service (`nlp/summarizer/service.py`)

The old `run()` is removed. `summarize(profile, fields)` is the single entry point. It is pure
dispatch: prepare the prompt fields (joining evidence lists, running the extractive reduction
when over budget), call the LLM with the profile's schema, validate + retry once if the profile
has a validator, and lift the profile's `output_keys`.

```python
import logging
import os
from pathlib import Path

from . import ollama_client, profiles
from .extractive import extract_top_sentences

log = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).parent / "prompts"
_TEMPLATE_CACHE: dict[str, str] = {}

# Single budget for the extractive pre-step. Its only purpose is to keep the prompt input
# within the LLM's context window, which is the same model for every profile — so there is one
# value, not one per profile. Expressed in words as a practical proxy for tokens.
_MAX_INPUT_WORDS = int(os.environ.get("SUMMARIZE_MAX_INPUT_WORDS", "400"))


def _template(prompt_file: str) -> str:
    if prompt_file not in _TEMPLATE_CACHE:
        _TEMPLATE_CACHE[prompt_file] = (_PROMPT_DIR / prompt_file).read_text(encoding="utf-8")
    return _TEMPLATE_CACHE[prompt_file]


def _reduce(text: str) -> str:
    """Extractive (TextRank) reduction, only when over the context-window budget."""
    if len(text.split()) <= _MAX_INPUT_WORDS:
        return text
    sentences = [s for s in text.replace("\n", " ").split(". ") if s.strip()]
    if len(sentences) <= 2:
        return text
    avg_words = sum(len(s.split()) for s in sentences) / len(sentences)
    n = max(1, int(_MAX_INPUT_WORDS / max(avg_words, 1)))
    return extract_top_sentences(text, n)


def _prepare_fields(profile, fields: dict) -> dict:
    """Normalise fields for the prompt: join evidence lists, apply extractive reduction."""
    prepared = dict(fields)
    # Profiles whose prompts consume an evidence list get a joined `evidence_text`.
    if "evidence" in prepared and isinstance(prepared["evidence"], list):
        prepared["evidence_text"] = "\n".join(
            f"{i + 1}. {e}" for i, e in enumerate(prepared["evidence"])
        )
    if "descriptions" in prepared and isinstance(prepared["descriptions"], list):
        prepared["descriptions_text"] = "\n".join(
            f"- {d}" for d in prepared["descriptions"]
        )
    prepared.setdefault("subtype", None)
    prepared["subtype_line"] = f"Subtipo: {prepared['subtype']}" if prepared.get("subtype") else ""
    # Extractive reduction on the profile's designated long field (context-window guard).
    if profile.extract_field and profile.extract_field in prepared:
        prepared[profile.extract_field] = _reduce(str(prepared[profile.extract_field]))
    return prepared


def summarize(profile_name: str, fields: dict) -> dict:
    """Generate text for the named profile. Returns the profile's output_keys.

    Raises ValueError for an unknown profile or malformed LLM JSON (after retries).
    Raises httpx.HTTPError if Ollama is unavailable.
    """
    profile = profiles.get(profile_name)
    prepared = _prepare_fields(profile, fields)
    prompt = _template(profile.prompt_file).format(**prepared)

    result = ollama_client.generate(prompt, schema=profile.schema)

    if profile.validate is not None:
        ok, reason = profile.validate(result)
        if not ok:
            log.info("validator rejected (%s); retry once with tightened prompt", reason)
            result = ollama_client.generate(prompt + profile.retry_suffix, schema=profile.schema)

    return {k: result[k] for k in profile.output_keys}
```

> `_reduce()` carries the old `_textrank_extract` logic (compute `n` from average sentence
> length, fall back to the original text when there are ≤2 sentences). The extractive step is
> therefore identical in behaviour to the retired `/extract` service, just internal and
> budget-gated.

---

## Ollama client change (`nlp/summarizer/ollama_client.py`)

`generate()` gains a `schema` parameter so each profile constrains decoding to its own output
shape. The module-level `_JSON_SCHEMA` becomes the default to preserve any direct caller.

```python
def generate(
    prompt: str,
    schema: dict = _JSON_SCHEMA,          # was hardcoded inside _call_once
    max_retries: int = 3,
    timeout: float = _TIMEOUT,
) -> dict[str, Any]:
    ...
```

`_call_once` takes the schema through and passes it as `"format": schema` (today it reads the
module global). No other change; retry/backoff behaviour is preserved.

```python
def _call_once(prompt: str, schema: dict, timeout: float) -> dict[str, Any]:
    response = httpx.post(
        f"{OLLAMA_HOST}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": schema},
        timeout=timeout,
    )
    response.raise_for_status()
    return json.loads(response.json()["response"])
```

---

## Prompt templates

**Prompts are domain-neutral task scaffolds.** The module stores the *task* (rewrite a title +
summarize, describe an entity/relation from evidence, merge descriptions) plus its output schema
and length rules — but **no use-case framing**. No "news", "journalism", "knowledge graph", or
financial/legal wording lives here; that would couple this agnostic module to one use case. If a
future use case needs a persona/context, add a `{context}` slot fed by the orchestrator
(hybrid) — not domain words baked into the module. (Extends the agnostic-module principle:
`[[project_schema_variable]]`.)

`prompts/article.es.txt` (renamed from `rewrite.es.txt`) — placeholders `{title}` (the original
headline to rewrite) and `{text}` (the body; reduced by the extractive step when over budget).
The article fields are `{title, text}`, used directly — no `title → headline` remapping.

```
Reescribe el titular y resume el contenido en español neutro.

REQUISITOS:
- El titular debe tener entre 8 y 15 palabras.
- Elimina del titular los sufijos de la fuente (por ejemplo " - dominio.com" o " | Sección") si aparecen.
- El resumen debe tener entre 2 y 4 frases.
- No añadas información que no esté en el texto.
- Devuelve únicamente JSON con los campos "headline" y "summary".

TITULAR ORIGINAL:
{title}

TEXTO:
{text}
```

`prompts/entity_desc.es.txt` (new):

```
Escribe una descripción concisa y factual de la entidad, basada únicamente en la evidencia proporcionada.

ENTIDAD
=======
Nombre: {name}
Tipo: {type}
{subtype_line}

EVIDENCIA
=========
{evidence_text}

REGLAS
======
- Máximo 2-4 frases.
- Afirma solo lo que la evidencia respalde explícitamente. No especules.
- Incluye los hechos y las relaciones que la evidencia confirme.
- Escribe la descripción en español, sea cual sea el idioma de la evidencia.
- Devuelve únicamente JSON con el campo "description".
```

`prompts/relation_desc.es.txt` (new): same shape, header block `RELACIÓN` with `{head}`,
`{tail}`, `{type}`, `{subtype_line}`, an `ATRIBUTOS` block for `{attributes}`, the same
`{evidence_text}` + neutral rules.

`prompts/aggregate.es.txt` (new): merges prior descriptions of the same element across sources.
Header block `{name}`/`{type}`, a `DESCRIPCIONES PREVIAS` block for `{descriptions_text}`, the
`{evidence_text}` block, and neutral merge rules (integrate, drop repeats/contradictions, no
speculation, Spanish).

---

## API (`api/routers/summarize.py`)

Generic, stateless, no embedding. `request_id` is an opaque echo for the caller's correlation.

```python
import logging
from fastapi import APIRouter, HTTPException
import httpx

from api.models import SummarizeRequest, SummarizeResponse
from api.warmth import mark_warm
from nlp.summarizer import service as summarizer_service

log = logging.getLogger(__name__)
router = APIRouter()


@router.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest) -> SummarizeResponse:
    try:
        result = summarizer_service.summarize(req.profile, req.fields)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except httpx.HTTPError as exc:
        log.error("ollama unavailable: %s", exc, extra={"request_id": req.request_id})
        raise HTTPException(status_code=503, detail="ollama_unavailable")
    except (KeyError, json_err()) as exc:        # json.JSONDecodeError
        log.error("ollama json format failed: %s", exc, extra={"request_id": req.request_id})
        raise HTTPException(status_code=503, detail="ollama_json_format_failed")
    mark_warm("summarize")
    return SummarizeResponse(request_id=req.request_id, result=result)
```

> `json_err()` stands in for `import json; json.JSONDecodeError` — keep the existing import.
> In-process orchestrators may skip the endpoint and call
> `summarizer_service.summarize(profile, fields)` directly; there is no hidden state.

### Models (`api/models.py`)

Replace the old `SummarizeRequest/Response`; delete `ExtractRequest/Response`.

```python
class SummarizeRequest(BaseModel):
    request_id: str | None = None
    profile:    str                 # "article" | "entity_desc" | "relation_desc" | "aggregate"
    fields:     dict                # profile-specific; validated by the profile/prompt

class SummarizeResponse(BaseModel):
    request_id: str | None = None
    result:     dict                # e.g. {"headline","summary"} or {"description"}
```

---

## Configuration

| Env var | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://ollama:11434` | Shared Ollama endpoint |
| `OLLAMA_MODEL` | `gemma4:e2b` | Generation model (all profiles, unless split later) |
| `OLLAMA_TIMEOUT` | `120` | Seconds per call |
| `SUMMARIZE_MAX_INPUT_WORDS` | `400` | Extractive pre-step reduces any profile's long input above this word count (single context-window guard) |

No new environment variables for model selection; if `entity_desc`/`aggregate` later need a
larger model, add a per-profile `model` field to `Profile` and thread it through `generate()`.

---

## Dependencies

- `sumy` — already used by the retired `nlp/extractor`; now a summarizer dependency (TextRank).
- Ollama sidecar — unchanged.
- `httpx` — already in requirements.
- No new packages. `nlp/encoder.py` (MiniLM) is no longer imported here — embedding is the
  `vectorizer` service's job (see spec §1, §3.7).
```
