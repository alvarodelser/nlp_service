"""Cross-document `merge` — the orchestrator operation, run in Phase B over dedup_corpus clusters.
Most-complete name wins; aliases/doc_ids unioned; evidence concatenated; attributes unioned with
highest-confidence non-null winning. Relations are never aggregated across distinct instances —
dedup_corpus only clusters relations judged the same ([[project_normalizer_edge_semantics]])."""
import json


def _union_attrs(members: list[dict], attr_names, attrs_of) -> dict:
    out: dict = {}
    for m in sorted(members, key=lambda m: m.get("confidence", 0.0), reverse=True):
        src = attrs_of(m)
        for k in attr_names:
            v = src.get(k)
            if v is not None and k not in out:
                out[k] = v
    return out


def _merge_entity_cluster(members: list[dict], attr_names) -> dict:
    best = max(members, key=lambda m: len(m["canonical_name"]))       # most complete name
    attrs = _union_attrs(members, attr_names, attrs_of=lambda m: m)
    return {
        "canonical_name": best["canonical_name"],
        "type": best["type"],
        "subtype": best.get("subtype"),
        "aliases": sorted({a for m in members for a in m.get("aliases", []) or []}),
        "doc_ids": sorted({d for m in members for d in m.get("doc_ids", []) or []}),
        "evidence": [e for m in members for e in m.get("evidence", []) or []],
        "confidence": max((m.get("confidence", 0.0) for m in members), default=0.0),
        **attrs,
    }


def _parse_attrs(m: dict) -> dict:
    raw = m.get("attributes")
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    return json.loads(raw)


def _merge_relation_cluster(members: list[dict], attr_names, canon: dict) -> dict:
    best = max(members, key=lambda m: m.get("confidence", 0.0))       # endpoints/type from best
    attrs = _union_attrs(members, attr_names, attrs_of=_parse_attrs)
    return {
        "head_id": canon.get(best["head_id"], best["head_id"]),
        "tail_id": canon.get(best["tail_id"], best["tail_id"]),
        "type": best["type"],
        "subtype": best.get("subtype"),
        "evidence": [e for m in members for e in m.get("evidence", []) or []],
        "doc_ids": sorted({d for m in members for d in m.get("doc_ids", []) or []}),
        "confidence": max((m.get("confidence", 0.0) for m in members), default=0.0),
        "attributes": attrs,
    }
