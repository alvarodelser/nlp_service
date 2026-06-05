# nlp_service/nlp/summarizer/profiles.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import validator

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


@dataclass(frozen=True)
class Profile:
    name:          str
    prompt_file:   str                         # under prompts/
    schema:        dict                        # Ollama output JSON schema
    output_keys:   tuple[str, ...]             # keys lifted from the LLM result
    extract_field: str | None                  # long input field reduced when over the budget
    validate:      Callable[[dict], tuple[bool, str | None]] | None = None
    retry_suffix:  str = ""


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
