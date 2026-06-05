# nlp_service/nlp/resolver/types.py
"""Resolver I/O contracts. In nlp/ so the service never imports api/."""
from pydantic import BaseModel


class EntityIn(BaseModel):
    name:     str
    type:     str
    subtype:  str | None = None
    evidence: str


class RelationIn(BaseModel):
    head:       str
    tail:       str
    type:       str
    subtype:    str | None = None
    evidence:   str
    attributes: dict = {}
    confidence: float


class ResolvedEntity(BaseModel):
    id:             str
    canonical_name: str
    names:          list[str]
    type:           str
    subtype:        str | None = None
    evidence:       list[str]


class ResolvedRelation(BaseModel):
    head_id:    str
    tail_id:    str
    type:       str
    subtype:    str | None = None
    evidence:   list[str]
    attributes: dict
    confidence: float


class ResolveResponse(BaseModel):
    entities:  list[ResolvedEntity]
    relations: list[ResolvedRelation]
