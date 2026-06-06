"""Deterministic place dedup for the news pipeline. The geotagger's places are already resolved,
so collapsing repeated mentions needs no LLM — a per-type identity key is sufficient. Not every
type carries the same id, and any place can be unresolved, so each key falls back to normalized
surface text, which keeps two distinct unresolved places from collapsing into one."""
import unicodedata


def _norm(s: str | None) -> str:
    s = (s or "").lower()
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn").strip()


def _place_key(p):
    t = p.type
    if t == "region":
        return ("region", p.geonames_id or _norm(p.name or p.text))
    if t == "city":
        return ("city", p.city_id or p.geonames_id or _norm(p.city_name or p.text))
    if t == "street":
        # a street is its geometry (edge_ids); unresolved → (imputed city, surface text)
        return ("street", tuple(sorted(p.edge_ids)) if p.edge_ids
                else (p.city_id, _norm(p.text)))
    # location: gazetteer point id, else rounded coords, else surface text
    return ("location", p.geonames_id
            or ((round(p.lat, 4), round(p.lon, 4)) if p.lat is not None else _norm(p.text)))


def dedupe_places(places) -> list:
    """Collapse repeated mentions deterministically, preserving first-seen order."""
    seen, out = set(), []
    for p in places:
        k = _place_key(p)
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out
