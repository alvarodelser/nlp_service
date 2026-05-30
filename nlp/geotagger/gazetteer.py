"""
City resolution and context-based disambiguation.

All spans must resolve to a city_id or be discarded — no raw coordinate output.
Disambiguation order:
  1. Direct GeoNames match above confidence threshold
  2. Dominant document city (most frequent high-confidence city across all spans)
  3. Sentence co-occurrence with an already-resolved span
  4. Source city prior
  5. Discard
"""

import json
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

import numpy as np

CITIES_PATH = os.getenv("CITIES_SNAPSHOT_PATH", "nlp/geotagger/data/cities_snapshot.json")
STREET_INDEX_PATH = os.getenv("STREET_INDEX_PATH", "nlp/geotagger/data/street_index.json")
SOURCE_PRIOR_PATH = os.getenv("SOURCE_CITY_PRIOR_PATH", "nlp/geotagger/data/source_city_prior.json")
GEONAMES_PATH = os.getenv("GEONAMES_PATH", "nlp/geotagger/data/geonames_es.tsv")

CONFIDENCE_THRESHOLD = 0.50
AMBIGUITY_THRESHOLD = 0.65

_cities: dict = {}           # name_normalised -> {city_id, city_name, bounds, ...}
_street_index: dict = {}     # city_id (str) -> {normalised_name: [edge_id, ...]}
_source_prior: dict = {}     # source_name -> city_id | null
_geonames: list[dict] = []   # [{name, feature_class, lat, lon, population, ...}]


def startup() -> None:
    global _cities, _street_index, _source_prior, _geonames
    if os.path.exists(CITIES_PATH):
        raw = json.loads(open(CITIES_PATH).read())
        _cities = {_normalise(c["city_name"]): c for c in raw}
    if os.path.exists(STREET_INDEX_PATH):
        _street_index = json.loads(open(STREET_INDEX_PATH).read())
    if os.path.exists(SOURCE_PRIOR_PATH):
        _source_prior = json.loads(open(SOURCE_PRIOR_PATH).read())
    if os.path.exists(GEONAMES_PATH):
        _geonames = _load_geonames(GEONAMES_PATH)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class CityResolution:
    city_id: int
    city_name: str
    confidence: float


@dataclass
class StreetResolution:
    span: str
    edge_ids: list[int]
    city_id: int


@dataclass
class PointResolution:
    span: str
    lat: float
    lon: float
    geonames_id: int | None


def resolve_cities(
    spans,              # list[Span] from ner.py
    source: str | None,
) -> tuple[list[CityResolution], list[StreetResolution], list[PointResolution]]:
    """
    Two-pass resolution:
      Pass B1: resolve each non-street span to a city.
      Pass B2: resolve street spans within winning city.
    Ambiguous spans go through the context cascade; if still unresolved,
    they are stored as geo_points (lat/lon from GeoNames) for map plotting.
    """
    source_city_id = _source_prior_city_id(source)

    # B1 — city resolution for non-street spans
    resolved: dict[str, CityResolution | None] = {}
    geonames_coords: dict[str, tuple[float, float, int | None]] = {}  # span → (lat, lon, id)
    for span in spans:
        if span.hint == "street":
            continue
        candidates = _lookup_city_candidates(span.text)
        resolution = _pick_candidate(candidates, source_city_id)
        resolved[span.text] = resolution
        # Cache the best GeoNames coordinate for fallback
        coords = _best_geonames_coords(span.text)
        if coords:
            geonames_coords[span.text] = coords

    # Determine dominant city from high-confidence resolutions
    high_conf = [r for r in resolved.values() if r and r.confidence >= AMBIGUITY_THRESHOLD]
    dominant_id = _dominant_city(high_conf)

    # Second pass: re-resolve ambiguous spans using dominant city + co-occurrence
    for span in spans:
        if span.hint == "street" or resolved.get(span.text) is not None:
            continue
        candidates = _lookup_city_candidates(span.text)
        if not candidates:
            continue
        if dominant_id:
            dc_candidates = [c for c in candidates if c.city_id == dominant_id]
            if dc_candidates:
                resolved[span.text] = dc_candidates[0]
                continue
        co_resolved = _cooccurrence_disambiguate(span, spans, resolved, candidates)
        resolved[span.text] = co_resolved  # None → falls back to geo_point

    geo_cities = _deduplicate_cities([r for r in resolved.values() if r is not None])

    # Spans that could not be resolved to a city → store as geo_points for map use
    geo_points: list[PointResolution] = []
    for span in spans:
        if span.hint == "street" or resolved.get(span.text) is not None:
            continue
        coords = geonames_coords.get(span.text)
        if coords:
            lat, lon, geonames_id = coords
            geo_points.append(PointResolution(span.text, lat, lon, geonames_id))

    # B2 — street resolution scoped to dominant city (or first resolved city)
    scope_city_id = dominant_id or (geo_cities[0].city_id if geo_cities else None)
    geo_streets: list[StreetResolution] = []
    if scope_city_id is not None:
        for span in spans:
            if span.hint != "street":
                continue
            street = _resolve_street(span.text, scope_city_id)
            if street:
                geo_streets.append(street)
            else:
                # Street unresolved via index — store GeoNames point if available
                coords = _best_geonames_coords(span.text)
                if coords:
                    lat, lon, geonames_id = coords
                    geo_points.append(PointResolution(span.text, lat, lon, geonames_id))

    return geo_cities, geo_streets, geo_points


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _lookup_city_candidates(name: str) -> list[CityResolution]:
    norm = _normalise(name)
    candidates = []

    # Direct snapshot lookup
    if norm in _cities:
        c = _cities[norm]
        candidates.append(CityResolution(c["city_id"], c["city_name"], 0.90))

    # GeoNames populated-place lookup
    for entry in _geonames:
        if entry["feature_class"] != "P":
            continue
        if _normalise(entry["name"]) == norm:
            pop_score = min(1.0, np.log1p(entry.get("population", 0)) / 15.0)
            candidates.append(CityResolution(
                entry.get("city_id", -1), entry["name"],
                0.50 + 0.35 * pop_score,
            ))

    return candidates


def _pick_candidate(
    candidates: list[CityResolution],
    source_city_id: int | None,
) -> CityResolution | None:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0] if candidates[0].confidence >= CONFIDENCE_THRESHOLD else None

    best = max(candidates, key=lambda c: c.confidence)
    if best.confidence >= AMBIGUITY_THRESHOLD:
        return best

    # Apply source prior as tiebreaker
    if source_city_id:
        matches = [c for c in candidates if c.city_id == source_city_id]
        if matches:
            return matches[0]

    # Ambiguous — will be resolved in second pass or discarded
    return None


def _dominant_city(resolutions: list[CityResolution]) -> int | None:
    if not resolutions:
        return None
    counts = Counter(r.city_id for r in resolutions)
    most_common_id, count = counts.most_common(1)[0]
    return most_common_id if count >= 1 else None


def _cooccurrence_disambiguate(
    span,
    all_spans,
    resolved: dict,
    candidates: list[CityResolution],
) -> CityResolution | None:
    sentence = span.sentence
    co_city_ids = {
        resolved[s.text].city_id
        for s in all_spans
        if s.text != span.text
        and s.sentence == sentence
        and resolved.get(s.text) is not None
    }
    for c in candidates:
        if c.city_id in co_city_ids:
            return c
    return None


def _resolve_street(span_text: str, city_id: int) -> StreetResolution | None:
    norm = _normalise_street(span_text)
    city_streets = _street_index.get(str(city_id), {})
    edge_ids = city_streets.get(norm)
    if edge_ids:
        return StreetResolution(span=span_text, edge_ids=edge_ids, city_id=city_id)
    return None


def _source_prior_city_id(source: str | None) -> int | None:
    if not source or source not in _source_prior:
        return None
    prior_city = _source_prior[source]
    if not prior_city:
        return None
    norm = _normalise(prior_city)
    city = _cities.get(norm)
    return city["city_id"] if city else None


def _deduplicate_cities(cities: list[CityResolution]) -> list[CityResolution]:
    seen: set[int] = set()
    result = []
    for c in sorted(cities, key=lambda x: -x.confidence):
        if c.city_id not in seen:
            seen.add(c.city_id)
            result.append(c)
    return result


def _normalise(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text.lower().strip())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _normalise_street(span: str) -> str:
    stripped = re.sub(
        r"^(Calle|Avda?\.?|Avenida|Plaza|Paseo|Ronda|Travesía|Carretera|C/)\s+",
        "",
        span.strip(),
        flags=re.IGNORECASE,
    )
    return _normalise(stripped)


def _best_geonames_coords(name: str) -> tuple[float, float, int | None] | None:
    norm = _normalise(name)
    for entry in _geonames:
        if _normalise(entry["name"]) == norm and entry.get("lat") and entry.get("lon"):
            return entry["lat"], entry["lon"], entry.get("geonames_id")
    return None


def _load_geonames(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 8:
                continue
            rows.append({
                "geonames_id": int(parts[0]) if parts[0].isdigit() else None,
                "name": parts[1],
                "feature_class": parts[6],
                "lat": float(parts[4]) if parts[4] else None,
                "lon": float(parts[5]) if parts[5] else None,
                "population": int(parts[14]) if len(parts) > 14 and parts[14] else 0,
            })
    return rows
