"""Orchestrator-owned ontology. There is no schema module in the NLP service: this loads and
validates the type catalogue (`config/ontologies/*.yaml`) and projects two artefacts — the inline
ExtractionSchema handed to `ner`, and the Weaviate collection specs (one column per declared
entity attribute; relation attributes kept as one JSON blob). `ner` only ever sees the projected
ExtractionSchema; it builds neither grammar source nor Weaviate spec."""
import os
import re

import yaml

from nlp.ner.schema_types import (AttributeDef, EntityTypeDef, ExtractionSchema,
                                   RelationTypeDef, SubtypeDef)

_DT = {"string": "text", "number": "number", "boolean": "boolean"}   # YAML datatype → Weaviate
_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")                          # SCREAMING_SNAKE


def _ontology_dir() -> str:
    return os.environ.get("ONTOLOGY_DIR", "config/ontologies")


def _check_name(name: str, what: str) -> None:
    if not _NAME_RE.match(name or ""):
        raise ValueError(f"{what} name must be SCREAMING_SNAKE_CASE: {name!r}")


def _check_attrs(attrs: dict, owner: str) -> None:
    for k, v in (attrs or {}).items():
        if v.get("type") not in _DT:
            raise ValueError(f"{owner} attribute {k!r} has invalid datatype {v.get('type')!r}")


def _validate(data: dict) -> None:
    entity_names = set()
    for e in data.get("entity_types", []):
        _check_name(e["name"], "entity type")
        entity_names.add(e["name"])
        for s in e.get("subtypes", []):
            _check_name(s["name"], "entity subtype")
        _check_attrs(e.get("attributes", {}), f"entity {e['name']}")
    if not entity_names:
        raise ValueError("ontology declares no entity types")

    for r in data.get("relation_types", []):
        _check_name(r["name"], "relation type")
        heads, tails = r.get("head_types"), r.get("tail_types")
        if not heads or not tails:
            raise ValueError(f"relation {r['name']!r} must declare head_types and tail_types")
        for t in [*heads, *tails]:
            if t not in entity_names:
                raise ValueError(f"relation {r['name']!r} references undeclared entity type {t!r}")
        _check_attrs(r.get("attributes", {}), f"relation {r['name']}")


def load(use_case: str) -> dict:
    with open(f"{_ontology_dir()}/{use_case}.yaml", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    _validate(data)
    return data


def extraction_schema(use_case: str) -> ExtractionSchema:
    data = load(use_case)
    return ExtractionSchema(
        entity_types=[EntityTypeDef(
            name=e["name"], description=e.get("description", ""),
            subtypes=[SubtypeDef(name=s["name"], description=s.get("description", ""))
                      for s in e.get("subtypes", [])],
            attributes=[AttributeDef(name=k, datatype=v["type"], description=v.get("description", ""))
                        for k, v in e.get("attributes", {}).items()],
        ) for e in data["entity_types"]],
        relation_types=[RelationTypeDef(
            name=r["name"], description=r.get("description", ""),
            subtypes=[SubtypeDef(name=s["name"], description=s.get("description", ""))
                      for s in r.get("subtypes", [])],
            head_types=r["head_types"], tail_types=r["tail_types"],
            attributes=[AttributeDef(name=k, datatype=v["type"], description=v.get("description", ""))
                        for k, v in r.get("attributes", {}).items()],
        ) for r in data["relation_types"]],
    )


def _dedupe_props(props: list[dict]) -> list[dict]:
    seen, out = set(), []
    for p in props:
        if p["name"] in seen:
            continue
        seen.add(p["name"])
        out.append(p)
    return out


def entity_attribute_names(use_case: str) -> list[str]:
    data = load(use_case)
    names: list[str] = []
    for e in data["entity_types"]:
        for k in e.get("attributes", {}):
            if k not in names:
                names.append(k)
    return names


def relation_attribute_names(use_case: str) -> list[str]:
    data = load(use_case)
    names: list[str] = []
    for r in data["relation_types"]:
        for k in r.get("attributes", {}):
            if k not in names:
                names.append(k)
    return names


def _suffix(final: bool) -> str:
    return "_final" if final else ""


def entity_collection_name(use_case: str, final: bool = False) -> str:
    return f"Entity_{use_case}{_suffix(final)}_v{load(use_case)['version']}"


def relation_collection_name(use_case: str, final: bool = False) -> str:
    return f"Relation_{use_case}{_suffix(final)}_v{load(use_case)['version']}"


def weaviate_entity_spec(use_case: str, final: bool = False) -> dict:
    data = load(use_case)
    attr_props = [{"name": k, "dataType": [_DT[v["type"]]]}
                  for e in data["entity_types"] for k, v in e.get("attributes", {}).items()]
    return {
        "class": entity_collection_name(use_case, final),
        "vectorizer": "none",
        "properties": _dedupe_props([
            {"name": "canonical_name", "dataType": ["text"]},
            {"name": "type", "dataType": ["text"]},
            {"name": "subtype", "dataType": ["text"]},
            {"name": "description", "dataType": ["text"]},
            {"name": "aliases", "dataType": ["text[]"]},
            {"name": "doc_ids", "dataType": ["text[]"]},
            {"name": "evidence", "dataType": ["text[]"]},
            {"name": "confidence", "dataType": ["number"]},
            *attr_props,
        ]),
        "vectorIndexConfig": {"distance": "cosine"},
    }


def weaviate_relation_spec(use_case: str, final: bool = False) -> dict:
    return {
        "class": relation_collection_name(use_case, final),
        "vectorizer": "none",
        "properties": [
            {"name": "head_id", "dataType": ["text"]},
            {"name": "tail_id", "dataType": ["text"]},
            {"name": "type", "dataType": ["text"]},
            {"name": "subtype", "dataType": ["text"]},
            {"name": "description", "dataType": ["text"]},
            {"name": "attributes", "dataType": ["text"]},     # JSON-encoded, opaque pass-through
            {"name": "evidence", "dataType": ["text[]"]},
            {"name": "doc_ids", "dataType": ["text[]"]},
            {"name": "confidence", "dataType": ["number"]},
        ],
        "vectorIndexConfig": {"distance": "cosine"},
    }
