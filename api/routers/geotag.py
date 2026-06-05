# nlp_service/api/routers/geotag.py
import logging
from dataclasses import asdict

from fastapi import APIRouter, HTTPException
import httpx

from api.models import GeoEntity, GeotagRequest, GeotagResponse
from api.warmth import mark_warm
from nlp.geotagger import service as geotagger_service

log = logging.getLogger(__name__)
router = APIRouter()


@router.post("/geotag", response_model=GeotagResponse)
def geotag(req: GeotagRequest) -> GeotagResponse:
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must be non-empty")
    try:
        result = geotagger_service.run(req.text, headline=req.headline, source=req.source)
    except FileNotFoundError as exc:
        log.error("geotagger data missing: %s", exc, extra={"request_id": req.request_id})
        raise HTTPException(status_code=503, detail="geotagger_data_missing")
    except httpx.HTTPError as exc:
        log.error("b4c cities API unavailable: %s", exc, extra={"request_id": req.request_id})
        raise HTTPException(status_code=503, detail="cities_api_unavailable")
    mark_warm("geotag")
    return GeotagResponse(
        request_id=req.request_id,
        places=[GeoEntity(**asdict(p)) for p in result["places"]],
    )
