# nlp_service/api/routers/dedup.py
import logging

from fastapi import APIRouter, HTTPException

import numpy as np

from api.models import (
    DedupRequest, DedupCheckEmbedRequest, DedupResponse,
    BootstrapRequest, BootstrapResponse,
)
from api.warmth import mark_warm
from nlp.dedup import service as dedup_service

log = logging.getLogger(__name__)

router = APIRouter()


@router.post("/dedup-check", response_model=DedupResponse)
def dedup_check(req: DedupRequest) -> DedupResponse:
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must be non-empty")
    result = dedup_service.check_minhash_only(req.article_id, req.text)
    mark_warm("dedup")
    return DedupResponse(
        article_id=req.article_id,
        duplicate_of=result["duplicate_of"],
        stage=result["stage"],
        score=result["score"],
        indexed=result["indexed"],
    )


@router.post("/dedup-check-embed", response_model=DedupResponse)
def dedup_check_embed(req: DedupCheckEmbedRequest) -> DedupResponse:
    vec = np.array(req.embedding_raw, dtype=np.float32)
    result = dedup_service.check_embedding_vec(req.article_id, vec)
    mark_warm("dedup")
    return DedupResponse(
        article_id=req.article_id,
        duplicate_of=result["duplicate_of"],
        stage=result["stage"],
        score=result["score"],
        indexed=result["indexed"],
    )


@router.post("/dedup/bootstrap", response_model=BootstrapResponse)
def dedup_bootstrap(req: BootstrapRequest) -> BootstrapResponse:
    articles = [{"article_id": a.article_id, "text": a.text} for a in req.articles]
    counts = dedup_service.bootstrap(articles)
    mark_warm("dedup")
    return BootstrapResponse(
        processed=counts["processed"],
        duplicates_found=counts["duplicates_found"],
        indexed=counts["indexed"],
    )
