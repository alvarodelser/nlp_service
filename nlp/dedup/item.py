# nlp_service/nlp/dedup/item.py
"""Single-item 2-step decision: similarity threshold, then LLM double-check in the mid band."""
from dataclasses import dataclass
from typing import Literal

from .llm import adjudicate


@dataclass
class ItemDecision:
    decision:   Literal["match", "no_match"]
    target_id:  str | None
    score:      float
    candidates: list


def decide_item(index, embedding, *, kind, compare_text, compare_property,
                match_threshold, llm_low, top_k, type_filter=None, llm=True) -> ItemDecision:
    cands = index.near(embedding, top_k, type_filter=type_filter,
                       return_props=(compare_property,))
    if not cands:
        return ItemDecision("no_match", None, 0.0, [])
    best = cands[0]                                          # nearVector returns best-first

    if best.score >= match_threshold:                       # step 1: high similarity → match
        return ItemDecision("match", best.id, best.score, cands)
    if best.score < llm_low or not llm:                     # below band → no match
        return ItemDecision("no_match", None, best.score, cands)

    verdict = adjudicate(kind, compare_text, best.props.get(compare_property, ""))
    decision = "match" if verdict == "yes" else "no_match"  # unsure → no_match (conservative)
    return ItemDecision(decision, best.id if decision == "match" else None, best.score, cands)
