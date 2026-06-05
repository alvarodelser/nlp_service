# nlp_service/nlp/geotagger/cities_api.py
"""HTTP client for the b4c cities API (city search + city-scoped edge search)."""
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
    """GET /cities/search?q= -> fuzzy city matches (each has at least id, name, slug)."""
    r = _get_client().get("/cities/search", params={"q": q})
    r.raise_for_status()
    return r.json().get("data", [])


def search_edges(city_id: int, q: str) -> list[dict]:
    """GET /cities/{city_id}/edges/search?q= -> fuzzy edge matches within that city."""
    r = _get_client().get(f"/cities/{city_id}/edges/search", params={"q": q})
    r.raise_for_status()
    return r.json().get("data", [])
