# Geotagger — Implementation Doc

**Module:** `nlp/geotagger/` · **Status:** Reworked
**Authoritative source:** `docs/superpowers/specs/2026-06-05-nlp-two-pipelines-design.md` §3.3

## Purpose

Pure **toponym detection → typing → resolution**. Given an article's text it returns a flat
list of resolved place entities. It makes **no scope decision** (the news orchestrator's `nli`
step owns city/regional/national — see doc 07) and writes no prose.

Pipeline inside the module:

```
text (+headline, +source)
   │
   ├─ detect spans            ner.extract_spans  (Flair LOC + street regex)   [unchanged]
   │
   ├─ type each span          street (regex) | region/city (gazetteer feature class)
   │                          | nli fallback when ambiguous or absent          [NEW]
   │
   ├─ resolve in order
   │    regions  → gazetteer (feature_class A)
   │    cities   → gazetteer (feature_class P) + b4c city id; nli tie-break on homonyms
   │    streets  → b4c cities API: search edges WITHIN detected cities          [NEW: HTTP API]
   │    locations→ gazetteer points (coords); impute nearest detected city
   │
   └─ places[]  (region | city | street | location)
```

Three structural changes drive this rework:

1. **Scope is gone.** `_classify_geo`, the scope hypotheses, the scope downgrade logic, and the
   `geo_scope`/`geo_region` outputs are **removed** — they move to the news orchestrator (doc 07).
2. **Streets come from the b4c cities API.** The local `config/street_index.json` (+ the
   `gazetteer` street functions) is **replaced** by HTTP calls to
   `https://wiig.dia.fi.upm.es/b4c_api/`. The API is city-scoped (find a city, then search its
   edges), so streets resolve **within the cities the article mentions**.
3. **City ids are b4c ids.** A resolved city is mapped to its b4c `id` via the API's city search,
   so `city_id` is consistent across cities, streets, and locations and the `edge_ids` reference
   the same system's geometry.

---

## What changes and what does not

| Component | Status |
|---|---|
| `nlp/geotagger/ner.py` | **Unchanged** (Flair NER + street regex) |
| `nlp/geotagger/gazetteer.py` — `load`, `lookup`, `_normalize`, source prior | **Kept** |
| `nlp/geotagger/gazetteer.py` — `load_streets`, `lookup_street`, `lookup_street_all_cities`, `_street_index`, `STREET_INDEX_PATH` | **Deleted** (→ b4c API) |
| `nlp/geotagger/cities_api.py` | **New** — httpx client for the b4c cities API |
| `nlp/geotagger/service.py` | **Rewritten** — detect → type → resolve → impute; no scope |
| `nlp/geotagger/data/geonames_es.tsv`, `data/cities_snapshot.json` | **Kept** (snapshot now only for source-prior name + centroids) |
| `config/street_index.json` | **Deleted** (data now behind the API) |
| `api/routers/geotag.py` | **Simplified** — new response, no scope |
| `api/models.py` — `GeotagResponse`, `GeoCity`, `GeoStreet`, `GeoPoint`, `PlaceMention` | **Replaced** by `GeoEntity` + new `GeotagResponse` |
| `api/main.py` warmup | `gazetteer.load_streets()` removed; geonames + snapshot warmup kept |

> The geotagger keeps importing `from nlp import nli` (doc 02) for typing + homonym tie-break.
> It never calls `nli.score()` and never decides scope.

---

## The b4c cities API (the contract `cities_api` depends on)

Base URL `https://wiig.dia.fi.upm.es/b4c_api/` (configurable). Both endpoints fuzzy-match
server-side (pg_trgm similarity when available, ILIKE fallback) and return `{data, message}`.

```
GET /cities/search?q=madr
  → { "data": [{ "id": 1, "name": "Madrid", "slug": "madrid", ... }],
      "message": "Found 1 city/cities matching 'madr'" }

GET /cities/1/edges/search?q=Alcalá
  → { "data": [ ...all edges named "Calle de Alcalá"... ],
      "message": "Found 14 edge(s) matching 'Alcalá'" }

GET /cities/1/edges/search?q=nonexistent
  → { "data": [], "message": "Found 0 edge(s) matching 'nonexistent'" }
```

Two-step flow: resolve a city name → `id` via `/cities/search`, then search that city's edges via
`/cities/{id}/edges/search`. Each edge object is assumed to carry an `id` (the edge/geometry id);
adjust the field name in `_edge_ids()` if the real payload differs.

---

## File layout (after rework)

```
nlp/geotagger/
  __init__.py
  ner.py            ← unchanged
  gazetteer.py      ← regions + cities (geonames) + source prior; street funcs removed
  cities_api.py     ← NEW: httpx client for the b4c cities API
  service.py        ← rewritten
  data/
    geonames_es.tsv
    cities_snapshot.json
api/routers/geotag.py
```

---

## Data contract

### Output — `GeoEntity` + `GeotagResponse` (`api/models.py`)

Replace `PlaceMention`, `GeoCity`, `GeoStreet`, `GeoPoint`, and the old `GeotagResponse`:

```python
class GeoEntity(BaseModel):
    text:        str                                   # surface span
    type:        Literal["region", "city", "street", "location"]
    name:        str | None = None                     # resolved canonical name
    geonames_id: int | None = None
    admin1_code: str | None = None                     # regions
    city_id:     int | None = None                     # b4c city id (city / street / location)
    city_name:   str | None = None
    edge_ids:    list[int] = []                         # streets only (geometry in the b4c DB)
    lat:         float | None = None                    # city / location only — never streets
    lon:         float | None = None
    confidence:  float = Field(default=1.0, ge=0.0, le=1.0)

class GeotagResponse(BaseModel):
    request_id: str | None = None
    places:     list[GeoEntity]
```

`GeotagRequest` keeps `text`, `headline`, `source`; rename `article_id` → optional `request_id`:

```python
class GeotagRequest(BaseModel):
    request_id: str | None = None
    text:       str
    headline:   str = ""
    source:     str = ""
```

No `geo_scope`/`geo_region`/`geo_cities`/`geo_points`/`all_places`/`city`/`city_confidence`.

> The existing `eval/` notebooks (test-only) are **deprecated**; a new test suite will be
> designed in a later pass and is out of scope for these implementation docs.

---

## b4c API client (`nlp/geotagger/cities_api.py`)

```python
import os
import httpx

_BASE = os.environ.get("B4C_API_BASE", "https://wiig.dia.fi.upm.es/b4c_api").rstrip("/")
_TIMEOUT = float(os.environ.get("B4C_API_TIMEOUT", "10"))
_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(base_url=_BASE, timeout=_TIMEOUT)
    return _client


def search_city(q: str) -> list[dict]:
    """GET /cities/search?q= → fuzzy city matches (each has at least id, name, slug)."""
    r = _get_client().get("/cities/search", params={"q": q})
    r.raise_for_status()
    return r.json().get("data", [])


def search_edges(city_id: int, q: str) -> list[dict]:
    """GET /cities/{city_id}/edges/search?q= → fuzzy edge matches within that city."""
    r = _get_client().get(f"/cities/{city_id}/edges/search", params={"q": q})
    r.raise_for_status()
    return r.json().get("data", [])
```

Network failures raise `httpx.HTTPError`; `service.run` lets them bubble so the router maps them
to `503` (a street simply stays unresolved only on an empty result, not on an outage).

---

## Typing (`service.py`) — type-first, nli fallback

```python
_TYPE_HYPOTHESES: list[tuple[str, str]] = [
    ("region",   "{} es una comunidad autónoma, provincia o región de España."),
    ("city",     "{} es una ciudad o municipio de España."),
    ("location", "{} es una calle, un edificio o un lugar concreto dentro de una ciudad."),
]


def _type_span(span: ner.Span, premise: str) -> str:
    # 1. Streets are pre-typed by the regex (high precision) — trust it.
    if span.hint == "street" or gazetteer._STREET_PREFIX_RE.match(span.text):
        return "street"
    # 2. Unambiguous gazetteer match settles region vs city without an LLM call.
    classes = {e.feature_class for e in gazetteer.lookup(span.text)}
    if classes == {"A"}:
        return "region"
    if classes == {"P"}:
        return "city"
    # 3. Ambiguous (both A and P, e.g. a province and its capital share a name) or absent from
    #    the gazetteer → ask nli to type it from article context.
    labels = [hyp.format(span.text) for _, hyp in _TYPE_HYPOTHESES]
    result = nli.classify(premise, labels=labels, multi_label=False)
    best_hyp = result["labels"][0]
    return next(t for t, hyp in _TYPE_HYPOTHESES if hyp.format(span.text) == best_hyp)
```

The gazetteer feature class handles the common case for free; `nli` only fires on genuine
ambiguity or out-of-gazetteer names.

---

## Resolution + imputation (`service.py`)

Regions and cities resolve **first** (so detected city centroids + b4c ids exist), then streets
and locations impute their city from those. One internal pass, no second round-trip.

```python
from __future__ import annotations
import json, logging, math, os, re, unicodedata
from pathlib import Path

from . import cities_api, gazetteer, ner
from nlp import nli

log = logging.getLogger("nlp_service.geotagger")

_CITIES_PATH = Path(os.environ.get(
    "CITIES_SNAPSHOT_PATH",
    Path(__file__).parent / "data" / "cities_snapshot.json"))
_cities: list[dict] = []
_by_id: dict[int, dict] = {}          # snapshot id → city dict (source-prior name lookup)

_CITY_TEMPLATE = "Este artículo describe principalmente hechos o iniciativas en {}."
_MAX_CITY_CANDIDATES = 10
_STREET_PREFIX_CI = re.compile(
    r'^(?:calle|avda?\.?|avenida|plaza|pza\.?|paseo|ps\.?|'
    r'glorieta|ronda|c/|camino|carretera|ctra\.?)\s+', re.IGNORECASE)

_b4c_city_cache: dict[str, dict | None] = {}    # normalized name → b4c city dict | None


def _normalize(s: str) -> str:
    s = s.lower()
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def _load_cities() -> None:
    global _cities, _by_id
    if _cities or not _CITIES_PATH.exists():
        return
    _cities = json.loads(_CITIES_PATH.read_text(encoding="utf-8"))
    for c in _cities:
        _by_id[c["id"]] = c


def load() -> None:
    gazetteer.load()
    gazetteer.load_source_prior()
    _load_cities()


def _haversine(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _b4c_city(name: str) -> dict | None:
    """Resolve a gazetteer city name to its b4c city record (cached). Prefers an exact match."""
    key = _normalize(name)
    if key not in _b4c_city_cache:
        results = cities_api.search_city(name)
        exact = next((c for c in results if _normalize(c["name"]) == key), None)
        _b4c_city_cache[key] = exact or (results[0] if results else None)
    return _b4c_city_cache[key]


def _street_query(text: str) -> str:
    """Drop the leading street-type prefix; keep accents/case for the API's fuzzy match."""
    return _STREET_PREFIX_CI.sub("", text).strip()


def _edge_ids(edges: list[dict]) -> list[int]:
    return [e["id"] for e in edges if "id" in e]


def _nearest_city_to_point(lat: float, lon: float, detected) -> "GeoEntity | None":
    """Nearest detected city to a concrete point (for locations, which have coordinates)."""
    best, best_d = None, float("inf")
    for d in detected:
        if d.lat is None or d.lon is None:
            continue
        dist = _haversine(lat, lon, d.lat, d.lon)
        if dist < best_d:
            best, best_d = d, dist
    return best


def run(text: str, headline: str = "", source: str = "") -> dict:
    load()
    premise = f"{headline}. {text}" if headline else text
    spans = ner.extract_spans(premise)
    typed = [(s, _type_span(s, premise)) for s in spans]

    places: list[GeoEntity] = []
    detected_cities: list[GeoEntity] = []

    for s, t in typed:                                   # regions
        if t == "region":
            places.append(_resolve_region(s))

    for s, t in typed:                                   # cities (+ collect detected)
        if t == "city":
            place = _resolve_city(s, premise)
            places.append(place)
            if place.city_id is not None and place.lat is not None:
                detected_cities.append(place)

    for s, t in typed:                                   # streets (search within detected cities)
        if t == "street":
            places.append(_resolve_street(s, source, detected_cities))

    for s, t in typed:                                   # locations (gazetteer coords)
        if t == "location":
            places.append(_resolve_location(s, detected_cities))

    return {"places": [p for p in places if p is not None]}
```

### Region resolution

```python
def _resolve_region(span: ner.Span) -> GeoEntity:
    entries = [e for e in gazetteer.lookup(span.text) if e.feature_class == "A"]
    if not entries:
        return GeoEntity(text=span.text, type="region")
    best = max(entries, key=lambda e: e.population)
    return GeoEntity(text=span.text, type="region", name=best.name,
                     geonames_id=best.geonames_id, admin1_code=best.admin1_code,
                     lat=best.lat, lon=best.lon)
```

### City resolution (gazetteer detection + b4c id; nli homonym tie-break)

```python
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
```

### Street resolution (b4c API, within detected cities)

```python
def _resolve_street(span: ner.Span, source: str, detected: list[GeoEntity]) -> GeoEntity:
    q = _street_query(span.text)

    # Candidate cities = the cities the article mentions (they carry b4c ids + centroids).
    candidates = [d for d in detected if d.city_id is not None]
    if not candidates and source:                         # no text context → source prior
        prior_id = gazetteer.get_city_prior(source)
        snap = _by_id.get(prior_id) if prior_id is not None else None
        b4c = _b4c_city(snap["name"]) if snap else None
        if b4c:
            candidates = [GeoEntity(text=snap["name"], type="city", city_id=b4c["id"],
                                    city_name=b4c["name"], lat=snap["lat"], lon=snap["lon"])]

    # Search the street within each candidate city; keep those that contain it.
    matches: list[tuple[GeoEntity, list[int]]] = []
    for c in candidates:
        edge_ids = _edge_ids(cities_api.search_edges(c.city_id, q))
        if edge_ids:
            matches.append((c, edge_ids))

    if not matches:
        return GeoEntity(text=span.text, type="street")   # unresolved; geometry unknown
    if len(matches) == 1:
        c, edge_ids = matches[0]
    else:
        # Street present in several mentioned cities: pick the one closest to the detected
        # cluster centroid (the city the article geographically centres on).
        pts = [(d.lat, d.lon) for d in detected if d.lat is not None]
        ref_lat = sum(p[0] for p in pts) / len(pts)
        ref_lon = sum(p[1] for p in pts) / len(pts)
        c, edge_ids = min(matches, key=lambda m: _haversine(m[0].lat, m[0].lon, ref_lat, ref_lon))

    return GeoEntity(text=span.text, type="street", city_id=c.city_id,
                     city_name=c.city_name, edge_ids=edge_ids)
```

> **Imputation (design round 3, adapted to the API):** the candidate cities are exactly the
> cities the article mentions — so a street is only ever attached to a city the text actually
> talks about ("the city is implicit in the text"). The b4c edge search confirms the geometry
> exists there. When several mentioned cities contain the same street name, the tie-break is the
> one closest to the detected-city cluster centroid. No mentioned city contains it → source
> prior; still nothing → `city_id = null`. Streets never carry coordinates (geometry is in the
> b4c DB, referenced by `edge_ids`).

### Location resolution (gazetteer coords + nearest detected city)

```python
def _resolve_location(span: ner.Span, detected: list[GeoEntity]) -> GeoEntity:
    points = [e for e in gazetteer.lookup(span.text) if e.feature_class == "P"]
    if not points:
        return GeoEntity(text=span.text, type="location")    # no gazetteer coords
    best = max(points, key=lambda e: e.population)
    near = _nearest_city_to_point(best.lat, best.lon, detected)   # nearest detected city to POI
    return GeoEntity(text=span.text, type="location", name=best.name,
                     geonames_id=best.geonames_id, lat=best.lat, lon=best.lon,
                     city_id=(near.city_id if near else None),
                     city_name=(near.city_name if near else None))
```

Coordinates come from the **gazetteer only** (design round 2). A location with no gazetteer match
is still returned (typed `location`, no coords) so the caller knows it was detected.

---

## API (`api/routers/geotag.py`)

```python
import logging
from fastapi import APIRouter, HTTPException
import httpx

from api.models import GeotagRequest, GeotagResponse
from api.warmth import mark_warm
from nlp.geotagger import service as geotagger_service

log = logging.getLogger(__name__)
router = APIRouter()


@router.post("/geotag", response_model=GeotagResponse)
def geotag(req: GeotagRequest) -> GeotagResponse:
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must be non-empty")
    try:
        result = geotagger_service.run(req.text, headline=req.headline, source=req.source)
    except FileNotFoundError as exc:
        log.error("geotagger data missing: %s", exc, extra={"request_id": req.request_id})
        raise HTTPException(status_code=503, detail="geotagger_data_missing")
    except httpx.HTTPError as exc:
        log.error("b4c cities API unavailable: %s", exc, extra={"request_id": req.request_id})
        raise HTTPException(status_code=503, detail="cities_api_unavailable")
    mark_warm("geotag")
    return GeotagResponse(request_id=req.request_id, places=result["places"])
```

`_resolve_*` return `GeoEntity` instances directly, so `result["places"]` is the response list.

---

## Configuration

| Env var | Default | Description |
|---|---|---|
| `B4C_API_BASE` | `https://wiig.dia.fi.upm.es/b4c_api` | Cities/edges API base URL |
| `B4C_API_TIMEOUT` | `10` | Seconds per API call |
| `CITIES_SNAPSHOT_PATH` | `nlp/geotagger/data/cities_snapshot.json` | City id/name/centroid snapshot (source-prior name lookup) |
| `SOURCE_PRIOR_PATH` | `config/source_city_prior.json` | Source → snapshot city_id fallback |
| `NER_DEVICE` | `cpu` | Flair device |
| `NLI_MODEL` | (see doc 02) | Shared XNLI model for typing + tie-break |

Removed: `STREET_INDEX_PATH` (streets now behind the b4c API).

---

## What moves out (scope) — nothing lost

The removed scope logic relocates to the news orchestrator (doc 07): the
city/regional/national pass and the region/city second pass are `nli.classify(...,
multi_label=False)` calls there, fed the `places[]` this module returns plus the source profile.
The old `_H_LOCAL/_H_REGIONAL/_H_NATIONAL` hypotheses and `_SCOPE_THRESHOLD` become orchestrator
config.

---

## Dependencies

- `httpx` — already in requirements (now also the b4c cities API client). **No Postgres driver.**
- `flair`, `torch` — already used by `ner.py`.
- `transformers` (via `nlp.nli`) — typing + tie-break.
- No FAISS, no embedding model here.
