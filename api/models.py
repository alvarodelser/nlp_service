# nlp_service/api/models.py
from pydantic import BaseModel, Field
from typing import Literal

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
    article_id: str
    text: str
    headline: str = ""
    source: str = ""


class PlaceMention(BaseModel):
    text: str
    type: Literal["city", "street", "region", "other"]
    lat: float | None = None
    lon: float | None = None
    geonames_id: int | None = None
    city_id: int | None = None


class GeoCity(BaseModel):
    city_id: int
    city_name: str
    confidence: float


class GeoStreet(BaseModel):
    span: str
    edge_ids: list[int]
    city_id: int | None = None


class GeoPoint(BaseModel):
    span: str
    lat: float
    lon: float
    geonames_id: int | None = None


class GeotagResponse(BaseModel):
    article_id: str
    geo_scope: Literal["national", "regional", "city"] | None = None
    geo_region: str | None = None
    geo_cities: list[GeoCity] = []
    geo_streets: list[GeoStreet] = []
    geo_points: list[GeoPoint] = []
    all_places: list[PlaceMention] = []
    # legacy — kept for backward compat with existing eval notebook
    city: str | None = None
    city_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


# --- Classify ---

class SourceProfile(BaseModel):
    city: str | None = None
    region: str | None = None
    topics: list[str] = []


class ClassifyRequest(BaseModel):
    article_id: str
    summary: str
    geo_cities: list[GeoCity] = []
    search_tags: list[str] = []
    source_profile: SourceProfile | None = None
    geo_scope: str | None = None


class ClassifyResponse(BaseModel):
    article_id: str
    topics: list[str]
    scores: dict[str, float]
    geo_scope: str  # national | regional | city
    out_of_scope: bool = False


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
