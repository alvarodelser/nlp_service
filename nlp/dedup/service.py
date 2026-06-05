# nlp_service/nlp/dedup/service.py
import os

from .corpus import cluster
from .item import decide_item
from .weaviate_index import WeaviateIndex

_ITEM_MATCH     = float(os.environ.get("DEDUP_ITEM_MATCH", "0.92"))
_ITEM_LLM_LOW   = float(os.environ.get("DEDUP_ITEM_LLM_LOW", "0.75"))
_CORPUS_MERGE   = float(os.environ.get("DEDUP_CORPUS_MERGE", "0.92"))
_CORPUS_LLM_LOW = float(os.environ.get("DEDUP_CORPUS_LLM_LOW", "0.80"))
_TOP_K          = int(os.environ.get("DEDUP_TOP_K", "10"))


def dedup_item(collection, embedding, *, kind="article", compare_text="",
               compare_property="summary", type_filter=None) -> dict:
    idx = WeaviateIndex(collection)
    d = decide_item(idx, embedding, kind=kind, compare_text=compare_text,
                    compare_property=compare_property,
                    match_threshold=_ITEM_MATCH, llm_low=_ITEM_LLM_LOW,
                    top_k=_TOP_K, type_filter=type_filter)
    return {"decision": d.decision, "target_id": d.target_id, "score": d.score,
            "candidates": [vars(c) for c in d.candidates]}


def dedup_corpus(collection, *, kind="entity", compare_property="description",
                 type_filter=None) -> dict:
    idx = WeaviateIndex(collection)
    clusters = cluster(idx, kind=kind, compare_property=compare_property,
                       merge_threshold=_CORPUS_MERGE, llm_low=_CORPUS_LLM_LOW,
                       top_k=_TOP_K, type_filter=type_filter)
    return {"clusters": clusters}
