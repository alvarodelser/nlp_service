from __future__ import annotations

import dataclasses
import json
import logging
import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import gazetteer, ner
from .ner import Span
from nlp import nli

log = logging.getLogger("nlp_service.geotagger")

_CITIES_PATH = Path(os.environ.get(
    "CITIES_SNAPSHOT_PATH",
    Path(__file__).parent.parent.parent / "nlp" / "geotagger" / "data" / "cities_snapshot.json",
))
_cities: list[dict] = []
_by_name: dict[str, dict] = {}

_H_LOCAL = (
    "Este artículo describe actuaciones, obras o iniciativas "
    "de un ayuntamiento o municipio concreto de España."
)
_H_REGIONAL = (
    "Este artículo describe políticas o actuaciones de una comunidad autónoma, "
    "diputación provincial o región de España que afectan a varios municipios."
)
_H_NATIONAL = (
    "Este artículo describe una política, ley, normativa o acontecimiento "
    "de alcance estatal en España, sin limitarse a una ciudad o región concreta."
)
_SCOPE_THRESHOLD = 0.35
_MAX_CITY_CANDIDATES = 10
_CITY_TEMPLATE = "Este artículo describe principalmente hechos o iniciativas en {}."
_REGION_TEMPLATE = (
    "Este artículo trata principalmente sobre "
    "la comunidad autónoma, provincia o región de {}."
)


@dataclass
class CityHit:
    city_name: str
    city_id: int
    lat: float
    lon: float
    geonames_id: int | None
    score: float


def _normalize(s: str) -> str:
    s = s.lower()
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def _load_cities() -> None:
    global _cities, _by_name
    if _cities:
        return
    if not _CITIES_PATH.exists():
        return
    _cities = json.loads(_CITIES_PATH.read_text(encoding="utf-8"))
    for c in _cities:
        _by_name[_normalize(c["name"])] = c
        if c.get("alt_name"):
            _by_name[_normalize(c["alt_name"])] = c


def _match_city(geo_entry: gazetteer.GeoEntry) -> dict | None:
    return _by_name.get(_normalize(geo_entry.name))


def load() -> None:
    gazetteer.load()
    gazetteer.load_streets()
    gazetteer.load_source_prior()
    _load_cities()


def _classify_geo(
    spans_with_geo: list[tuple[str, list[gazetteer.GeoEntry]]],
    premise: str,
) -> tuple[str, CityHit | None, str | None]:
    city_pool: list[tuple[gazetteer.GeoEntry, dict]] = []
    region_pool: list[gazetteer.GeoEntry] = []
    seen_city_ids: set[int] = set()
    seen_region_names: set[str] = set()

    for _span_text, entries in spans_with_geo:
        for entry in entries:
            if entry.feature_class == "P":
                city = _match_city(entry)
                if city and city["id"] not in seen_city_ids:
                    city_pool.append((entry, city))
                    seen_city_ids.add(city["id"])
            elif entry.feature_class == "A":
                if entry.name not in seen_region_names:
                    region_pool.append(entry)
                    seen_region_names.add(entry.name)

    scope_result = nli.classify(
        premise,
        labels=[_H_LOCAL, _H_REGIONAL, _H_NATIONAL],
        multi_label=False,
    )
    score_map = dict(zip(scope_result["labels"], scope_result["scores"]))
    local_s = score_map.get(_H_LOCAL, 0.0)
    regional_s = score_map.get(_H_REGIONAL, 0.0)
    national_s = score_map.get(_H_NATIONAL, 0.0)
    best_score = max(local_s, regional_s, national_s)

    if best_score < _SCOPE_THRESHOLD:
        geo_scope = "city" if city_pool else ("regional" if region_pool else "national")
    elif local_s >= regional_s and local_s >= national_s:
        geo_scope = "city"
    elif regional_s >= local_s and regional_s >= national_s:
        geo_scope = "regional"
    else:
        geo_scope = "national"

    city_hit: CityHit | None = None
    geo_region: str | None = None

    if geo_scope == "city":
        city_hit = _pick_city(city_pool, premise)
        if city_hit is None and not city_pool:
            # NLI scored city scope but no city candidates in gazetteer.
            # Downgrade to regional or national based on what was actually detected.
            geo_scope = "regional" if region_pool else "national"
            if geo_scope == "regional":
                geo_region = _pick_region(region_pool, premise)
    elif geo_scope == "regional":
        geo_region = _pick_region(region_pool, premise)

    return geo_scope, city_hit, geo_region


def _pick_city(
    city_pool: list[tuple[gazetteer.GeoEntry, dict]],
    premise: str,
) -> CityHit | None:
    if not city_pool:
        return None
    if len(city_pool) == 1:
        entry, city = city_pool[0]
        return CityHit(
            city_name=city["name"], city_id=city["id"],
            lat=entry.lat, lon=entry.lon,
            geonames_id=entry.geonames_id, score=1.0,
        )
    candidates = sorted(city_pool, key=lambda x: x[0].population, reverse=True)
    candidates = candidates[:_MAX_CITY_CANDIDATES]
    city_names = [city["name"] for _, city in candidates]

    result = nli.classify(premise, labels=city_names, multi_label=False,
                          hypothesis_template=_CITY_TEMPLATE)
    best_name = result["labels"][0]
    best_score = result["scores"][0]

    matched = next(((e, c) for e, c in candidates if c["name"] == best_name), None)
    if matched is None:
        return None
    entry, city = matched
    return CityHit(
        city_name=city["name"], city_id=city["id"],
        lat=entry.lat, lon=entry.lon,
        geonames_id=entry.geonames_id, score=best_score,
    )


def _pick_region(region_pool: list[gazetteer.GeoEntry], premise: str) -> str | None:
    if not region_pool:
        return None
    if len(region_pool) == 1:
        return region_pool[0].name
    region_names = [e.name for e in region_pool]
    result = nli.classify(premise, labels=region_names, multi_label=False,
                          hypothesis_template=_REGION_TEMPLATE)
    return result["labels"][0]


def run(
    text: str,
    headline: str = "",
    source: str = "",
) -> dict[str, Any]:
    load()

    full_input = f"{headline}. {text}" if headline else text
    premise = full_input

    spans = ner.extract_spans(full_input)
    street_spans = [s for s in spans if s.hint == "street"]
    loc_spans = [s for s in spans if s.hint != "street"]

    log.info("Stage A — loc=%s street=%s",
             [s.text for s in loc_spans], [s.text for s in street_spans])

    source_prior_city_id = gazetteer.get_city_prior(source) if source else None
    spans_with_geo = [(s.text, gazetteer.lookup(s.text)) for s in loc_spans]

    log.info("Stage B1 — gazetteer hits: %s",
             {t: [e.name for e in entries] for t, entries in spans_with_geo})

    rescued: list[ner.Span] = [
        s for s, entries in zip(loc_spans, [e for _, e in spans_with_geo])
        if not entries and gazetteer._STREET_PREFIX_RE.match(s.text)
    ]
    for r in rescued:
        loc_spans = [s for s in loc_spans if s is not r]
        spans_with_geo = [(t, e) for t, e in spans_with_geo if not (t == r.text and not e)]
        street_spans.append(dataclasses.replace(r, hint="street"))

    geo_scope, city_hit, geo_region = _classify_geo(spans_with_geo, premise)

    log.info("Stage B2 — scope=%s city=%s region=%s",
             geo_scope,
             city_hit.city_name if city_hit else None,
             geo_region)

    winning_city_id = city_hit.city_id if city_hit else None

    geo_streets: list[dict] = []
    for s in street_spans:
        if winning_city_id:
            edge_ids = gazetteer.lookup_street(winning_city_id, s.text)
            if edge_ids:
                geo_streets.append({"span": s.text, "edge_ids": edge_ids,
                                    "city_id": winning_city_id})
                continue
        matches = gazetteer.lookup_street_all_cities(s.text)
        if len(matches) == 1:
            cid, edge_ids = next(iter(matches.items()))
            geo_streets.append({"span": s.text, "edge_ids": edge_ids, "city_id": cid})
        elif len(matches) > 1 and source_prior_city_id and source_prior_city_id in matches:
            geo_streets.append({"span": s.text,
                                 "edge_ids": matches[source_prior_city_id],
                                 "city_id": source_prior_city_id})
        else:
            geo_streets.append({"span": s.text, "edge_ids": [], "city_id": None})

    geo_points: list[dict] = []
    for _span_text, entries in spans_with_geo:
        if city_hit:
            continue
        p_entries = [e for e in entries if e.feature_class == "P"]
        if p_entries:
            best = max(p_entries, key=lambda x: x.population)
            geo_points.append({
                "span": _span_text, "lat": best.lat, "lon": best.lon,
                "geonames_id": best.geonames_id,
            })

    geo_cities: list[dict] = []
    if city_hit:
        geo_cities = [{
            "city_id": city_hit.city_id,
            "city_name": city_hit.city_name,
            "confidence": city_hit.score,
        }]

    all_places: list[dict] = []
    for span_text, entries in spans_with_geo:
        if not entries:
            all_places.append({"text": span_text, "type": "other",
                                "lat": None, "lon": None,
                                "geonames_id": None, "city_id": None})
            continue
        best = max(entries, key=lambda x: x.population)
        ptype = ("city" if best.feature_class == "P"
                 else ("region" if best.feature_class == "A" else "other"))
        cid = city_hit.city_id if (city_hit and best.geonames_id == city_hit.geonames_id) else None
        all_places.append({
            "text": span_text, "type": ptype,
            "lat": best.lat, "lon": best.lon,
            "geonames_id": best.geonames_id, "city_id": cid,
        })
    for s in street_spans:
        all_places.append({"text": s.text, "type": "street",
                            "lat": None, "lon": None,
                            "geonames_id": None, "city_id": winning_city_id})

    log.info("Stage B3 — streets=%s points=%s",
             [s["span"] for s in geo_streets], [p["span"] for p in geo_points])

    return {
        "geo_scope": geo_scope,
        "geo_region": geo_region,
        "geo_cities": geo_cities,
        "geo_streets": geo_streets,
        "geo_points": geo_points,
        "all_places": all_places,
        "city": city_hit.city_name if city_hit else None,
        "city_confidence": city_hit.score if city_hit else 0.0,
    }
