# nlp_service/api/routers/nli.py
import logging

from fastapi import APIRouter, HTTPException

from api.models import NliRequest, NliResponse
from api.warmth import mark_warm
from nlp import nli

log = logging.getLogger(__name__)

router = APIRouter()


@router.post("/nli", response_model=NliResponse)
def run_nli(req: NliRequest) -> NliResponse:
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must be non-empty")
    if not req.hypotheses:
        raise HTTPException(status_code=422, detail="hypotheses must be non-empty")
    scores = nli.score(
        text=req.text,
        hypotheses=req.hypotheses,
        threshold=req.threshold,
        blacklist=req.blacklist,
        hypothesis_template=req.hypothesis_template,
    )
    mark_warm("nli")
    return NliResponse(request_id=req.request_id, scores=scores)
