# nlp_service/nlp/ner/schema_types.py
"""The inline ExtractionSchema contract (shared with the orchestrator) and the NER output
contract. Lives in nlp/ so the service never imports api/."""
from pydantic import BaseModel
from typing import Literal


# --- Inline schema (orchestrator builds it; ner consumes it) ---

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
    subtypes:   list[SubtypeDef] = []
    attributes: list[AttributeDef] = []


class RelationTypeDef(BaseModel):
    name:       str
    description: str = ""
    subtypes:   list[SubtypeDef] = []
    attributes: list[AttributeDef] = []
    head_types: list[str]
    tail_types: list[str]


class ExtractionSchema(BaseModel):
    entity_types:   list[EntityTypeDef]
    relation_types: list[RelationTypeDef]


# --- Output contract ---

class ExtractedEntity(BaseModel):
    name:          str
    mention_text:  str
    type:          str
    subtype:       str | None = None
    evidence_text: str
    attributes:    dict[str, str | float | bool] = {}
    confidence:    float


class ExtractedRelation(BaseModel):
    head:          int
    tail:          int
    type:          str
    subtype:       str | None = None
    evidence_text: str
    attributes:    dict[str, str | float | bool] = {}
    confidence:    float


class NerResponse(BaseModel):
    entities:  list[ExtractedEntity]
    relations: list[ExtractedRelation]
