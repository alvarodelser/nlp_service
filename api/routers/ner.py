# nlp_service/api/routers/ner.py
import logging

from fastapi import APIRouter, HTTPException
import httpx

from api.models import NerRequest
from api.warmth import mark_warm
from nlp.ner import service as ner_service
from nlp.ner.schema_types import NerResponse

log = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ner", response_model=NerResponse)
def ner(req: NerRequest) -> NerResponse:
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must be non-empty")
    try:
        result = ner_service.run(req.text, req.extraction_schema)
    except httpx.HTTPError as exc:
        log.error("ollama unavailable: %s", exc, extra={"request_id": req.request_id})
        raise HTTPException(status_code=503, detail="ollama_unavailable")
    mark_warm("ner")
    return result
