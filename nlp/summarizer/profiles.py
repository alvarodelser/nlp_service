# nlp_service/nlp/summarizer/profiles.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import formula, validator

_ARTICLE_SCHEMA = {
    "type": "object",
    "properties": {"headline": {"type": "string"}, "summary": {"type": "string"}},
    "required": ["headline", "summary"],
}
_FORMULA_SCHEMA = {
    "type": "object",
    "properties": {
        "has_formula": {"type": "boolean"},
        "literal":     {"type": "boolean"},          # copied from the text (true) or deduced from words (false)
        "expression":  {"type": "string"},
        "variables":   {"type": "array", "items": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}, "meaning": {"type": "string"},
                           "kind": {"type": "string", "enum": ["output", "input", "parameter", "constant"]},
                           "value": {"type": ["number", "null"]}, "unit": {"type": "string"}},
            "required": ["symbol", "meaning", "kind", "value", "unit"]}},
        "conditions":  {"type": "array", "items": {"type": "string"}},
        "source_text": {"type": "string"},
    },
    "required": ["has_formula", "literal", "expression", "variables", "conditions", "source_text"],
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
    # Domain profiles: the fields (attributes_text) go in alongside the evidence, and the rules
    # forbid paraphrasing figures, which is where a generic description does harm.
    "requirement_desc": Profile(
        name="requirement_desc",
        prompt_file="requirement_desc.es.txt",
        schema=_DESCRIPTION_SCHEMA,
        output_keys=("description",),
        extract_field="evidence_text",
    ),
    "criterion_desc": Profile(
        name="criterion_desc",
        prompt_file="criterion_desc.es.txt",
        schema=_DESCRIPTION_SCHEMA,
        output_keys=("description",),
        extract_field="evidence_text",
    ),
    # A scoring formula as a parseable expression with every symbol explained. No extractive
    # reduction: the formula is often one line that TextRank would drop; the caller caps the text.
    "criterion_formula": Profile(
        name="criterion_formula",
        prompt_file="criterion_formula.es.txt",
        schema=_FORMULA_SCHEMA,
        output_keys=("has_formula", "literal", "expression", "variables", "conditions", "source_text"),
        extract_field=None,
        validate=formula.check,
        retry_suffix="\nRECUERDA: «SALIDA = expresión» con la sintaxis indicada y cada símbolo declarado en variables.",
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
