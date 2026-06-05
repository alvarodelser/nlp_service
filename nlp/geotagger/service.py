from __future__ import annotations

import logging
import math
import re
import unicodedata
from dataclasses import dataclass, field

from . import cities_api, gazetteer, ner
from nlp import nli

log = logging.getLogger("nlp_service.geotagger")

_CITY_TEMPLATE = "Este artículo describe principalmente hechos o iniciativas en {}."
_MAX_CITY_CANDIDATES = 10
_STREET_PREFIX_CI = re.compile(
    r'^(?:calle|avda?\.?|avenida|plaza|pza\.?|paseo|ps\.?|'
    r'glorieta|ronda|c/|camino|carretera|ctra\.?)\s+', re.IGNORECASE)

_TYPE_HYPOTHESES: list[tuple[str, str]] = [
    ("region",   "{} es una comunidad autónoma, provincia o región de España."),
    ("city",     "{} es una ciudad o municipio de España."),
    ("location", "{} es una calle, un edificio o un lugar concreto dentro de una ciudad."),
]

_b4c_city_cache: dict[str, dict | None] = {}    # normalized name -> b4c city dict | None


@dataclass
class GeoEntity:
    text:        str
    type:        str                  # region | city | street | location
    name:        str | None = None
    geonames_id: int | None = None
    admin1_code: str | None = None
    city_id:     int | None = None
    city_name:   str | None = None
    edge_ids:    list[int] = field(default_factory=list)
    lat:         float | None = None
    lon:         float | None = None
    confidence:  float = 1.0


def _normalize(s: str) -> str:
    s = s.lower()
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def load() -> None:
    gazetteer.load()


# --- geometry helpers ---

def _haversine(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _nearest_city_to_point(lat: float, lon: float, detected: list[GeoEntity]) -> GeoEntity | None:
    best, best_d = None, float("inf")
    for d in detected:
        if d.lat is None or d.lon is None:
            continue
        dist = _haversine(lat, lon, d.lat, d.lon)
        if dist < best_d:
            best, best_d = d, dist
    return best


# --- b4c city id + street query helpers ---

def _b4c_city(name: str) -> dict | None:
    """Resolve a gazetteer city name to its b4c city record (cached). Prefers an exact match."""
    key = _normalize(name)
    if key not in _b4c_city_cache:
        try:
            results = cities_api.search_city(name)
        except Exception as exc:                 # noqa: BLE001 — city id is best-effort
            log.warning("b4c city search failed for %r: %s", name, exc)
            return None                          # do not cache transient failures
        exact = next((c for c in results if _normalize(c["name"]) == key), None)
        _b4c_city_cache[key] = exact or (results[0] if results else None)
    return _b4c_city_cache[key]


def _street_query(text: str) -> str:
    """Drop the leading street-type prefix; keep accents/case for the API's fuzzy match."""
    return _STREET_PREFIX_CI.sub("", text).strip()


def _edge_ids(edges: list[dict]) -> list[int]:
    return [e["id"] for e in edges if "id" in e]


# --- typing ---

def _type_span(span: ner.Span, premise: str) -> str:
    if span.hint == "street" or gazetteer._STREET_PREFIX_RE.match(span.text):
        return "street"
    classes = {e.feature_class for e in gazetteer.lookup(span.text)}
    if classes == {"A"}:
        return "region"
    if classes == {"P"}:
        return "city"
    labels = [hyp.format(span.text) for _, hyp in _TYPE_HYPOTHESES]
    result = nli.classify(premise, labels=labels, multi_label=False)
    best_hyp = result["labels"][0]
    return next(t for t, hyp in _TYPE_HYPOTHESES if hyp.format(span.text) == best_hyp)


# --- resolution ---

def _resolve_region(span: ner.Span) -> GeoEntity | None:
    entries = [e for e in gazetteer.lookup(span.text) if e.feature_class == "A"]
    if not entries:
        return None                                   # not in gazetteer → drop
    best = max(entries, key=lambda e: e.population)
    return GeoEntity(text=span.text, type="region", name=best.name,
                     geonames_id=best.geonames_id, admin1_code=best.admin1_code,
                     lat=best.lat, lon=best.lon)


def _resolve_city(span: ner.Span, premise: str) -> GeoEntity | None:
    entries = [e for e in gazetteer.lookup(span.text) if e.feature_class == "P"]
    if not entries:
        return None                                   # not in gazetteer → drop
    if len(entries) == 1:
        best, conf = entries[0], 1.0
    else:
        cands = sorted(entries, key=lambda e: e.population, reverse=True)[:_MAX_CITY_CANDIDATES]
        result = nli.classify(premise, labels=[e.name for e in cands],
                              multi_label=False, hypothesis_template=_CITY_TEMPLATE)
        best = next(e for e in cands if e.name == result["labels"][0])
        conf = float(result["scores"][0])
    b4c = _b4c_city(best.name)
    if not b4c:
        return None                                   # no b4c city id → drop
    return GeoEntity(text=span.text, type="city", name=best.name,
                     geonames_id=best.geonames_id, city_id=b4c["id"], city_name=b4c["name"],
                     lat=best.lat, lon=best.lon, confidence=conf)


def _resolve_street(span: ner.Span, detected: list[GeoEntity]) -> GeoEntity | None:
    q = _street_query(span.text)
    matches: list[tuple[GeoEntity, list[int]]] = []
    for c in detected:                                # only the cities the article mentions
        edge_ids = _edge_ids(cities_api.search_edges(c.city_id, q))
        if edge_ids:
            matches.append((c, edge_ids))

    if not matches:
        return None                                   # not found in any mentioned city → drop
    if len(matches) == 1:
        c, edge_ids = matches[0]
    else:                                             # closest to the detected-city cluster
        pts = [(d.lat, d.lon) for d in detected if d.lat is not None]
        ref_lat = sum(p[0] for p in pts) / len(pts)
        ref_lon = sum(p[1] for p in pts) / len(pts)
        c, edge_ids = min(matches, key=lambda m: _haversine(m[0].lat, m[0].lon, ref_lat, ref_lon))

    return GeoEntity(text=span.text, type="street", city_id=c.city_id,
                     city_name=c.city_name, edge_ids=edge_ids)


def _resolve_location(span: ner.Span, detected: list[GeoEntity]) -> GeoEntity | None:
    points = [e for e in gazetteer.lookup(span.text) if e.feature_class == "P"]
    if not points:
        return None                                   # no gazetteer coords → drop
    best = max(points, key=lambda e: e.population)
    near = _nearest_city_to_point(best.lat, best.lon, detected)
    return GeoEntity(text=span.text, type="location", name=best.name,
                     geonames_id=best.geonames_id, lat=best.lat, lon=best.lon,
                     city_id=(near.city_id if near else None),
                     city_name=(near.city_name if near else None))


def run(text: str, headline: str = "", debug: bool = False) -> dict:
    load()
    premise = f"{headline}. {text}" if headline else text
    spans = ner.extract_spans(premise)
    typed = [(s, _type_span(s, premise)) for s in spans]

    places: list[GeoEntity] = []
    detected_cities: list[GeoEntity] = []

    for s, t in typed:
        if t == "region":
            place = _resolve_region(s)
            if place:
                places.append(place)

    for s, t in typed:
        if t == "city":
            place = _resolve_city(s, premise)
            if place:
                places.append(place)
                if place.lat is not None:
                    detected_cities.append(place)

    for s, t in typed:
        if t == "street":
            place = _resolve_street(s, detected_cities)
            if place:
                places.append(place)

    for s, t in typed:
        if t == "location":
            place = _resolve_location(s, detected_cities)
            if place:
                places.append(place)

    result = {"places": places}
    if debug:
        result["trace"] = {
            # stage 1 — raw NER + regex detection
            "spans": [{"text": s.text, "label": s.label, "hint": s.hint,
                       "start": s.start_char, "end": s.end_char} for s in spans],
            # stage 2 — type assigned to each span (regex / gazetteer / nli fallback)
            "typed": [{"text": s.text, "type": t} for s, t in typed],
            # stage 3 context — cities used to impute streets/locations
            "detected_cities": [{"city_id": d.city_id, "city_name": d.city_name,
                                 "lat": d.lat, "lon": d.lon} for d in detected_cities],
        }
    return result
