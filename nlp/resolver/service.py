# nlp_service/nlp/resolver/service.py
import logging
from collections import defaultdict

import httpx

from . import ollama_client
from .edges import rewrite_relations
from .premerge import Candidate, premerge
from .types import EntityIn, RelationIn, ResolvedEntity, ResolveResponse

log = logging.getLogger(__name__)


def _refine(candidates: list[Candidate]) -> list[dict]:
    """Cluster candidates. Refines per type (the only valid merges are within a type, which also
    bounds the LLM input). Falls back to singletons when the LLM is unavailable or omits a
    candidate. Returns clusters: {canonical_name, type, subtype, candidate_indices(global)}."""
    by_type: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(candidates):
        by_type[c.type].append(i)

    clusters: list[dict] = []
    for typ, idxs in by_type.items():
        if len(idxs) == 1:
            c = candidates[idxs[0]]
            clusters.append({"canonical_name": c.canonical_name, "type": typ,
                             "subtype": c.subtype, "candidate_indices": idxs})
            continue

        local = [candidates[i] for i in idxs]
        try:
            refined = ollama_client.refine(local)
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            log.warning("resolver refine failed for type %s; using singletons: %s", typ, exc)
            refined = [{"canonical_name": c.canonical_name, "subtype": c.subtype,
                        "candidate_indices": [li]} for li, c in enumerate(local)]

        seen: set[int] = set()
        for cl in refined:
            local_idx = [li for li in cl.get("candidate_indices", []) if 0 <= li < len(local)]
            if not local_idx:
                continue
            seen.update(local_idx)
            clusters.append({"canonical_name": cl["canonical_name"], "type": typ,
                             "subtype": cl.get("subtype"),
                             "candidate_indices": [idxs[li] for li in local_idx]})
        for li in range(len(local)):          # any candidate the LLM dropped → singleton
            if li not in seen:
                c = local[li]
                clusters.append({"canonical_name": c.canonical_name, "type": typ,
                                 "subtype": c.subtype, "candidate_indices": [idxs[li]]})
    return clusters


def resolve(entities, relations) -> ResolveResponse:
    """Intra-document resolution: reduce mentions to canonical entities (with ids) and rewrite
    relations onto them. Accepts EntityIn/RelationIn or plain dicts."""
    entities = [e if isinstance(e, EntityIn) else EntityIn(**e) for e in entities]
    relations = [r if isinstance(r, RelationIn) else RelationIn(**r) for r in relations]
    if not entities and not relations:
        return ResolveResponse(entities=[], relations=[])

    candidates = premerge(entities)
    clusters = _refine(candidates)

    resolved_entities: list[ResolvedEntity] = []
    name_to_id: dict[str, str] = {}
    for i, cluster in enumerate(clusters):
        cid = str(i)
        members = [candidates[idx] for idx in cluster["candidate_indices"]]
        names: list[str] = []
        evidence: list[str] = []
        for m in members:
            for n in m.names:
                if n not in names:
                    names.append(n)
            evidence.extend(m.evidence)
        resolved_entities.append(ResolvedEntity(
            id=cid, canonical_name=cluster["canonical_name"], names=names,
            type=cluster["type"], subtype=cluster.get("subtype"), evidence=evidence))
        for n in names:
            if n in name_to_id:
                log.warning("name %r maps to multiple entities; keeping first (%s)", n, name_to_id[n])
            else:
                name_to_id[n] = cid

    resolved_relations = rewrite_relations(relations, name_to_id)
    return ResolveResponse(entities=resolved_entities, relations=resolved_relations)
