from pydantic import BaseModel


class ExtractRequest(BaseModel):
    article_id: str
    text: str


class ExtractResponse(BaseModel):
    extract: str
    embedding_raw: list[float]


class DedupCheckRequest(BaseModel):
    article_id: str
    text: str


class DedupCheckEmbedRequest(BaseModel):
    article_id: str
    embedding_raw: list[float]


class DedupResponse(BaseModel):
    duplicate_of: str | None


class SummarizeRequest(BaseModel):
    article_id: str
    text: str
    extract: str
    headline: str


class SummarizeResponse(BaseModel):
    headline: str
    summary: str
    embedding_summary: list[float]


class GeotagRequest(BaseModel):
    article_id: str
    text: str
    headline: str


class ResolvedCity(BaseModel):
    city_id: int
    city_name: str
    confidence: float


class ResolvedStreet(BaseModel):
    span: str
    edge_ids: list[int]
    city_id: int


class GeotagResponse(BaseModel):
    geo_cities: list[ResolvedCity]
    geo_streets: list[ResolvedStreet]


class SourceProfile(BaseModel):
    city: str | None = None
    region: str | None = None
    topics: list[str] = []


class ClassifyRequest(BaseModel):
    article_id: str
    summary: str
    geo_cities: list[ResolvedCity]
    search_tags: list[str] = []
    source_profile: SourceProfile | None = None


class ClassifyResponse(BaseModel):
    topics: list[str]
    scores: dict[str, float]
    geo_scope: str  # national | regional | city
