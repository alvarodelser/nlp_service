# nlp_service/nlp/resolver/edges.py
"""Pure-Python relation rewrite + overlap collapse. No LLM. Relations are never aggregated:
distinct evidence stays distinct; only exact-evidence duplicates (the same sentence from two
overlapping chunks) collapse."""
import logging

from .types import RelationIn, ResolvedRelation

log = logging.getLogger(__name__)


def _fingerprint(evidence: str) -> str:
    return " ".join(evidence.lower().split())


def rewrite_relations(relations: list[RelationIn], name_to_id: dict[str, str]) -> list[ResolvedRelation]:
    groups: dict[tuple, ResolvedRelation] = {}
    order: list[tuple] = []
    for r in relations:
        head_id = name_to_id.get(r.head)
        tail_id = name_to_id.get(r.tail)
        if head_id is None or tail_id is None:
            log.warning("dropping relation %r: endpoint not resolved (%r -> %r)",
                        r.type, r.head, r.tail)
            continue
        key = (head_id, r.type, tail_id, _fingerprint(r.evidence))
        existing = groups.get(key)
        if existing is None:
            rr = ResolvedRelation(head_id=head_id, tail_id=tail_id, type=r.type, subtype=r.subtype,
                                  evidence=[r.evidence], attributes=dict(r.attributes),
                                  confidence=r.confidence)
            groups[key] = rr
            order.append(key)
        else:
            existing.evidence.append(r.evidence)
            if r.confidence > existing.confidence:        # higher-confidence instance wins
                existing.confidence = r.confidence
                existing.attributes = dict(r.attributes)  # attributes never aggregated
    return [groups[k] for k in order]
