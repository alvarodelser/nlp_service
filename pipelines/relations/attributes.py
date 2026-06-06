"""Attribute join. The resolver deliberately never threads attributes (EntityIn/ResolvedEntity
carry none), per [[project_normalizer_edge_semantics]] / [[project_schema_variable]]. The
orchestrator therefore keeps its own name -> (attributes, confidence) index built from the ner
ExtractedEntity list, and joins it onto each ResolvedEntity via its surface names at upsert:
highest-confidence non-null wins, restricted to the declared attribute columns. Attributes stay
opaque pass-throughs — no domain rules here."""
from collections import defaultdict


def build_attribute_index(ner_entities) -> dict:
    """ner_entities: iterable of {name, attributes, confidence}. Returns name -> list of
    (confidence, attributes), one entry per mention."""
    idx: dict[str, list[tuple[float, dict]]] = defaultdict(list)
    for e in ner_entities:
        idx[e["name"]].append((float(e.get("confidence", 0.0)), e.get("attributes", {}) or {}))
    return idx


def join_entity_attributes(names, index, attr_names) -> dict:
    """Union the declared attributes across all of a resolved entity's names; for each attribute,
    the highest-confidence non-null mention wins."""
    records: list[tuple[float, dict]] = []
    for n in names:
        records.extend(index.get(n, []))
    records.sort(key=lambda r: r[0], reverse=True)        # highest confidence first
    out: dict = {}
    for _conf, attrs in records:
        for k in attr_names:
            v = attrs.get(k)
            if v is not None and k not in out:
                out[k] = v
    return out
