# nlp_service/api/routers/summarize.py
import json
import logging

from fastapi import APIRouter, HTTPException
import httpx

from api.models import SummarizeRequest, SummarizeResponse
from api.warmth import mark_warm
from nlp.summarizer import service as summarizer_service

log = logging.getLogger(__name__)

router = APIRouter()


@router.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest) -> SummarizeResponse:
    try:
        result = summarizer_service.summarize(req.profile, req.fields)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except httpx.HTTPError as exc:
        log.error("ollama unavailable: %s", exc, extra={"request_id": req.request_id})
        raise HTTPException(status_code=503, detail="ollama_unavailable")
    except (KeyError, json.JSONDecodeError) as exc:
        log.error("ollama json format failed: %s", exc, extra={"request_id": req.request_id})
        raise HTTPException(status_code=503, detail="ollama_json_format_failed")
    mark_warm("summarize")
    return SummarizeResponse(request_id=req.request_id, result=result)
