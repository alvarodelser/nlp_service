# nlp_service/api/models.py
from pydantic import BaseModel, Field
from typing import Literal

from nlp.ner.schema_types import ExtractionSchema

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


# --- NER ---

class NerRequest(BaseModel):
    # `extraction_schema` is exposed over HTTP as "schema" (avoids shadowing BaseModel.schema).
    model_config = {"populate_by_name": True}

    request_id:        str | None = None
    text:              str
    extraction_schema: ExtractionSchema = Field(alias="schema")


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

class DedupRequest(BaseModel):
    article_id: str
    text: str


class DedupCheckEmbedRequest(BaseModel):
    article_id: str
    embedding_raw: list[float]


class DedupResponse(BaseModel):
    article_id: str
    duplicate_of: str | None
    stage: Literal["minhash", "embedding"] | None = None
    score: float | None = None
    indexed: bool


class BootstrapArticle(BaseModel):
    article_id: str
    text: str


class BootstrapRequest(BaseModel):
    articles: list[BootstrapArticle]


class BootstrapResponse(BaseModel):
    processed: int
    duplicates_found: int
    indexed: int
