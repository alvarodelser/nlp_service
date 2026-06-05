# nlp_service/api/routers/dedup.py
import logging

from fastapi import APIRouter, HTTPException
import httpx

from api.models import (DedupItemRequest, DedupItemResponse,
                        DedupCorpusRequest, DedupCorpusResponse)
from api.warmth import mark_warm
from nlp.dedup import service as dedup_service

log = logging.getLogger(__name__)

router = APIRouter()


@router.post("/dedup/item", response_model=DedupItemResponse)
def dedup_item(req: DedupItemRequest) -> DedupItemResponse:
    try:
        result = dedup_service.dedup_item(
            req.collection, req.embedding, kind=req.kind, compare_text=req.compare_text,
            compare_property=req.compare_property, type_filter=req.type_filter)
    except httpx.HTTPError as exc:
        log.error("weaviate unavailable: %s", exc, extra={"request_id": req.request_id})
        raise HTTPException(status_code=503, detail="weaviate_unavailable")
    mark_warm("dedup")
    return DedupItemResponse(request_id=req.request_id, **result)


@router.post("/dedup/corpus", response_model=DedupCorpusResponse)
def dedup_corpus(req: DedupCorpusRequest) -> DedupCorpusResponse:
    try:
        result = dedup_service.dedup_corpus(
            req.collection, kind=req.kind, compare_property=req.compare_property,
            type_filter=req.type_filter)
    except httpx.HTTPError as exc:
        log.error("weaviate unavailable: %s", exc, extra={"request_id": req.request_id})
        raise HTTPException(status_code=503, detail="weaviate_unavailable")
    mark_warm("dedup")
    return DedupCorpusResponse(request_id=req.request_id, **result)
