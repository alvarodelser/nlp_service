from __future__ import annotations

import json
import logging
import math
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from . import cities_api, gazetteer, ner
from nlp import nli

log = logging.getLogger("nlp_service.geotagger")

_CITIES_PATH = Path(os.environ.get(
    "CITIES_SNAPSHOT_PATH",
    str(Path(__file__).parent / "data" / "cities_snapshot.json"),
))
_cities: list[dict] = []
_by_id: dict[int, dict] = {}          # snapshot id -> city dict (source-prior name lookup)

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


def _load_cities() -> None:
    global _cities, _by_id
    if _cities:
        return
    if not _CITIES_PATH.exists():
        return
    _cities = json.loads(_CITIES_PATH.read_text(encoding="utf-8"))
    for c in _cities:
        _by_id[c["id"]] = c


def load() -> None:
    gazetteer.load()
    gazetteer.load_source_prior()
    _load_cities()


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

def _resolve_region(span: ner.Span) -> GeoEntity:
    entries = [e for e in gazetteer.lookup(span.text) if e.feature_class == "A"]
    if not entries:
        return GeoEntity(text=span.text, type="region")
    best = max(entries, key=lambda e: e.population)
    return GeoEntity(text=span.text, type="region", name=best.name,
                     geonames_id=best.geonames_id, admin1_code=best.admin1_code,
                     lat=best.lat, lon=best.lon)


def _resolve_city(span: ner.Span, premise: str) -> GeoEntity:
    entries = [e for e in gazetteer.lookup(span.text) if e.feature_class == "P"]
    if not entries:
        return GeoEntity(text=span.text, type="city")
    if len(entries) == 1:
        best, conf = entries[0], 1.0
    else:
        cands = sorted(entries, key=lambda e: e.population, reverse=True)[:_MAX_CITY_CANDIDATES]
        result = nli.classify(premise, labels=[e.name for e in cands],
                              multi_label=False, hypothesis_template=_CITY_TEMPLATE)
        best = next(e for e in cands if e.name == result["labels"][0])
        conf = float(result["scores"][0])
    b4c = _b4c_city(best.name)
    return GeoEntity(text=span.text, type="city", name=best.name,
                     geonames_id=best.geonames_id,
                     city_id=(b4c["id"] if b4c else None),
                     city_name=(b4c["name"] if b4c else best.name),
                     lat=best.lat, lon=best.lon, confidence=conf)


def _resolve_street(span: ner.Span, source: str, detected: list[GeoEntity]) -> GeoEntity:
    q = _street_query(span.text)

    candidates = [d for d in detected if d.city_id is not None]
    if not candidates and source:
        prior_id = gazetteer.get_city_prior(source)
        snap = _by_id.get(prior_id) if prior_id is not None else None
        b4c = _b4c_city(snap["name"]) if snap else None
        if b4c:
            candidates = [GeoEntity(text=snap["name"], type="city", city_id=b4c["id"],
                                    city_name=b4c["name"], lat=snap.get("lat"),
                                    lon=snap.get("lon"))]

    matches: list[tuple[GeoEntity, list[int]]] = []
    for c in candidates:
        try:
            edge_ids = _edge_ids(cities_api.search_edges(c.city_id, q))
        except Exception as exc:                 # noqa: BLE001 — unresolved is acceptable
            log.warning("b4c edge search failed (city=%s, q=%r): %s", c.city_id, q, exc)
            continue
        if edge_ids:
            matches.append((c, edge_ids))

    if not matches:
        return GeoEntity(text=span.text, type="street")
    if len(matches) == 1:
        c, edge_ids = matches[0]
    else:
        pts = [(d.lat, d.lon) for d in detected if d.lat is not None and d.lon is not None]
        ref_lat = sum(p[0] for p in pts) / len(pts)
        ref_lon = sum(p[1] for p in pts) / len(pts)
        c, edge_ids = min(matches, key=lambda m: _haversine(m[0].lat, m[0].lon, ref_lat, ref_lon))

    return GeoEntity(text=span.text, type="street", city_id=c.city_id,
                     city_name=c.city_name, edge_ids=edge_ids)


def _resolve_location(span: ner.Span, detected: list[GeoEntity]) -> GeoEntity:
    points = [e for e in gazetteer.lookup(span.text) if e.feature_class == "P"]
    if not points:
        return GeoEntity(text=span.text, type="location")
    best = max(points, key=lambda e: e.population)
    near = _nearest_city_to_point(best.lat, best.lon, detected)
    return GeoEntity(text=span.text, type="location", name=best.name,
                     geonames_id=best.geonames_id, lat=best.lat, lon=best.lon,
                     city_id=(near.city_id if near else None),
                     city_name=(near.city_name if near else None))


def run(text: str, headline: str = "", source: str = "") -> dict:
    load()
    premise = f"{headline}. {text}" if headline else text
    spans = ner.extract_spans(premise)
    typed = [(s, _type_span(s, premise)) for s in spans]

    places: list[GeoEntity] = []
    detected_cities: list[GeoEntity] = []

    for s, t in typed:
        if t == "region":
            places.append(_resolve_region(s))

    for s, t in typed:
        if t == "city":
            place = _resolve_city(s, premise)
            places.append(place)
            if place.city_id is not None and place.lat is not None:
                detected_cities.append(place)

    for s, t in typed:
        if t == "street":
            places.append(_resolve_street(s, source, detected_cities))

    for s, t in typed:
        if t == "location":
            places.append(_resolve_location(s, detected_cities))

    return {"places": [p for p in places if p is not None]}
