import numpy as np
from fastapi import APIRouter
from api.models import DedupCheckRequest, DedupCheckEmbedRequest, DedupResponse
from nlp.dedup import service as dedup_svc

router = APIRouter()


@router.post("/dedup-check", response_model=DedupResponse)
def dedup_check(req: DedupCheckRequest) -> DedupResponse:
    duplicate_of = dedup_svc.check_minhash(req.article_id, req.text)
    return DedupResponse(duplicate_of=duplicate_of)


@router.post("/dedup-check-embed", response_model=DedupResponse)
def dedup_check_embed(req: DedupCheckEmbedRequest) -> DedupResponse:
    embedding = np.array(req.embedding_raw, dtype=np.float32)
    duplicate_of = dedup_svc.check_embedding(req.article_id, embedding)
    return DedupResponse(duplicate_of=duplicate_of)
