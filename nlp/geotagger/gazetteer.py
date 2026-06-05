# nlp_service/nlp/geotagger/gazetteer.py
from __future__ import annotations

import csv
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

_STREET_PREFIX_RE = re.compile(
    r'^(?:calle|avda?\.?|avenida|plaza|pza\.?|paseo|ps\.?|'
    r'glorieta|ronda|c/|camino|carretera|ctra\.?)\s+',
    re.IGNORECASE,
)

_DATA_PATH = Path(__file__).parent / "data" / "geonames_es.tsv"
_entries: dict[str, list["GeoEntry"]] = {}


@dataclass
class GeoEntry:
    geonames_id: int
    name: str
    lat: float
    lon: float
    feature_class: str
    feature_code: str
    admin1_code: str
    population: int


def _normalize(s: str) -> str:
    s = s.lower()
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def load() -> None:
    if _entries:
        return
    with _DATA_PATH.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                e = GeoEntry(
                    geonames_id=int(row["geonames_id"]),
                    name=row["name"],
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    feature_class=row["feature_class"],
                    feature_code=row["feature_code"],
                    admin1_code=row["admin1_code"],
                    population=int(row["population"] or 0),
                )
            except (ValueError, KeyError):
                continue
            for key in {_normalize(row["name"]), _normalize(row["asciiname"])}:
                _entries.setdefault(key, []).append(e)


def lookup(span_text: str) -> list[GeoEntry]:
    load()
    return _entries.get(_normalize(span_text), [])


# Streets now resolve via the b4c cities API (nlp/geotagger/cities_api.py), not a local index.
_SOURCE_PRIOR_PATH = Path(os.environ.get(
    "SOURCE_PRIOR_PATH",
    Path(__file__).parent.parent.parent / "config" / "source_city_prior.json",
))

_source_prior: dict = {}                               # {source_name: city_id or None}


def load_source_prior() -> None:
    global _source_prior
    if _source_prior:
        return
    if _SOURCE_PRIOR_PATH.exists():
        _source_prior = json.loads(_SOURCE_PRIOR_PATH.read_text(encoding="utf-8"))


def get_city_prior(source_name: str) -> int | None:
    """Return city_id prior for a known source, or None."""
    return _source_prior.get(source_name)
