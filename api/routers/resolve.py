# nlp_service/api/routers/resolve.py
import logging

from fastapi import APIRouter, HTTPException
import httpx

from api.models import ResolveRequest
from api.warmth import mark_warm
from nlp.resolver import service as resolver_service
from nlp.resolver.types import ResolveResponse

log = logging.getLogger(__name__)

router = APIRouter()


@router.post("/resolve", response_model=ResolveResponse)
def resolve(req: ResolveRequest) -> ResolveResponse:
    try:
        result = resolver_service.resolve(req.entities, req.relations)
    except httpx.HTTPError as exc:
        log.error("ollama unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="ollama_unavailable")
    mark_warm("resolve")
    return result
