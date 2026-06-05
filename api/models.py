# nlp_service/api/models.py
from pydantic import BaseModel, Field
from typing import Literal

from nlp.ner.schema_types import ExtractionSchema
from nlp.resolver.types import EntityIn, RelationIn

# --- Summarize ---

class SummarizeRequest(BaseModel):
    request_id: str | None = None
    profile:    str                 # "article" | "entity_desc" | "relation_desc" | "aggregate"
    fields:     dict                # profile-specific; validated by the profile/prompt


class SummarizeResponse(BaseModel):
    request_id: str | None = None
    result:     dict                # e.g. {"headline","summary"} or {"description"}


# --- Geotag ---

class GeotagRequest(BaseModel):
    request_id: str | None = None
    text:       str
    headline:   str = ""
    source:     str = ""
    debug:      bool = False        # include per-stage `trace` in the response


class GeoEntity(BaseModel):
    text:        str
    type:        Literal["region", "city", "street", "location"]
    name:        str | None = None
    geonames_id: int | None = None
    admin1_code: str | None = None
    city_id:     int | None = None
    city_name:   str | None = None
    edge_ids:    list[int] = []
    lat:         float | None = None
    lon:         float | None = None
    confidence:  float = Field(default=1.0, ge=0.0, le=1.0)


class GeotagResponse(BaseModel):
    request_id: str | None = None
    places:     list[GeoEntity]
    trace:      dict | None = None      # per-stage internals when debug=True


# --- NER ---

class NerRequest(BaseModel):
    # `extraction_schema` is exposed over HTTP as "schema" (avoids shadowing BaseModel.schema).
    model_config = {"populate_by_name": True}

    request_id:        str | None = None
    text:              str
    extraction_schema: ExtractionSchema = Field(alias="schema")


# --- Resolve ---

class ResolveRequest(BaseModel):
    entities:  list[EntityIn]
    relations: list[RelationIn]


# --- NLI ---

class NliRequest(BaseModel):
    request_id:          str | None = None
    text:                str
    hypotheses:          list[str]
    threshold:           float | None = None
    blacklist:           bool = False
    hypothesis_template: str = "{}"


class ScorePair(BaseModel):
    hypothesis: str
    score:      float = Field(ge=0.0, le=1.0)


class NliResponse(BaseModel):
    request_id: str | None = None
    scores:     list[ScorePair]      # input order; len < len(hypotheses) ⇒ short-circuited


# --- Dedup ---

class DedupItemRequest(BaseModel):
    request_id:       str | None = None
    collection:       str
    embedding:        list[float]
    kind:             str = "article"      # article | entity | relation (LLM prompt flavour)
    compare_text:     str = ""             # new item's comparable text (for the LLM step)
    compare_property: str = "summary"      # stored property read for candidates
    type_filter:      str | None = None    # restrict candidates by `type`


class Candidate(BaseModel):
    id:    str
    score: float
    props: dict


class DedupItemResponse(BaseModel):
    request_id: str | None = None
    decision:   Literal["match", "no_match"]
    target_id:  str | None
    score:      float
    candidates: list[Candidate]


class DedupCorpusRequest(BaseModel):
    request_id:       str | None = None
    collection:       str
    kind:             str = "entity"
    compare_property: str = "description"
    type_filter:      str | None = None


class DedupCorpusResponse(BaseModel):
    request_id: str | None = None
    clusters:   list[list[str]]            # each inner list = one cluster of object ids
