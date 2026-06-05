# nlp_service/nlp/ner/schema_compiler.py
"""Compile an ExtractionSchema into (JSON-Schema grammar, prompt). Pure, cached by schema hash.

The grammar is a discriminated union (one branch per type) so the model cannot emit an undeclared
type, a foreign attribute, a wrong subtype, or a wrong datatype. A leading `reasoning` string is a
scratchpad (constrained decoding otherwise forbids chain-of-thought)."""
import hashlib
import json
from pathlib import Path

from .schema_types import ExtractionSchema

_PROMPT_PATH = Path(__file__).parent / "prompts" / "extraction.txt"
_TEMPLATE: str | None = None
_cache: dict[str, tuple[dict, str]] = {}

_DT = {
    "string":  {"type": "string"},
    "number":  {"type": "number"},
    "boolean": {"type": "boolean"},
}


def _template() -> str:
    global _TEMPLATE
    if _TEMPLATE is None:
        _TEMPLATE = _PROMPT_PATH.read_text(encoding="utf-8")
    return _TEMPLATE


def _attr_props(attrs) -> dict:
    return {a.name: _DT[a.datatype] for a in attrs}


def _subtype_schema(subtypes) -> dict:
    names = [s.name for s in subtypes]
    if not names:
        return {"type": "null"}
    return {"anyOf": [{"type": "string", "enum": names}, {"type": "null"}]}


def _entity_branch(et) -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["name", "mention_text", "type", "subtype",
                     "evidence_text", "attributes", "confidence"],
        "properties": {
            "name":          {"type": "string"},
            "mention_text":  {"type": "string"},
            "type":          {"const": et.name},
            "subtype":       _subtype_schema(et.subtypes),
            "evidence_text": {"type": "string"},
            "attributes":    {"type": "object", "additionalProperties": False,
                              "properties": _attr_props(et.attributes)},
            "confidence":    {"type": "number"},
        },
    }


def _relation_branch(rt) -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["head", "tail", "type", "subtype",
                     "evidence_text", "attributes", "confidence"],
        "properties": {
            "head":          {"type": "integer"},
            "tail":          {"type": "integer"},
            "type":          {"const": rt.name},
            "subtype":       _subtype_schema(rt.subtypes),
            "evidence_text": {"type": "string"},
            "attributes":    {"type": "object", "additionalProperties": False,
                              "properties": _attr_props(rt.attributes)},
            "confidence":    {"type": "number"},
        },
    }


def _array_of(branches: list[dict]) -> dict:
    if not branches:
        return {"type": "array", "maxItems": 0}
    return {"type": "array", "items": {"oneOf": branches}}


def _build_grammar(schema: ExtractionSchema) -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["reasoning", "entities", "relations"],
        "properties": {
            "reasoning": {"type": "string"},
            "entities":  _array_of([_entity_branch(e) for e in schema.entity_types]),
            "relations": _array_of([_relation_branch(r) for r in schema.relation_types]),
        },
    }


def _build_prompt(schema: ExtractionSchema) -> str:
    lines = ["ENTIDADES"]
    for et in schema.entity_types:
        lines.append(f"- {et.name}: {et.description}".rstrip())
        for s in et.subtypes:
            lines.append(f"    subtipo {s.name}: {s.description}".rstrip())
        for a in et.attributes:
            lines.append(f"    atributo {a.name} ({a.datatype}): {a.description}".rstrip())
    lines.append("RELACIONES")
    for rt in schema.relation_types:
        lines.append(f"- {rt.name}: {rt.description} "
                     f"[sujeto: {', '.join(rt.head_types)}; objeto: {', '.join(rt.tail_types)}]".rstrip())
        for a in rt.attributes:
            lines.append(f"    atributo {a.name} ({a.datatype}): {a.description}".rstrip())
    return _template().format(type_catalogue="\n".join(lines))


def compile_schema(schema: ExtractionSchema) -> tuple[dict, str]:
    key = hashlib.sha1(
        json.dumps(schema.model_dump(), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    if key not in _cache:
        _cache[key] = (_build_grammar(schema), _build_prompt(schema))
    return _cache[key]
