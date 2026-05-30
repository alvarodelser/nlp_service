from nlp.geotagger import ner, gazetteer
from api.models import GeotagResponse, ResolvedCity, ResolvedStreet


def startup() -> None:
    ner.startup()
    gazetteer.startup()


def geotag(text: str, headline: str, source: str | None = None) -> GeotagResponse:
    spans = ner.extract_spans(text, headline)
    city_resolutions, street_resolutions = gazetteer.resolve_cities(spans, source)

    return GeotagResponse(
        geo_cities=[
            ResolvedCity(
                city_id=r.city_id,
                city_name=r.city_name,
                confidence=round(r.confidence, 3),
            )
            for r in city_resolutions
        ],
        geo_streets=[
            ResolvedStreet(
                span=r.span,
                edge_ids=r.edge_ids,
                city_id=r.city_id,
            )
            for r in street_resolutions
        ],
    )
