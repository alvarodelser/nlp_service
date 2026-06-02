# nlp_service/api/routers/classify.py
import logging

from fastapi import APIRouter, HTTPException

from api.models import ClassifyRequest, ClassifyResponse
from api.warmth import mark_warm
from nlp.classifier import service as classifier_service

log = logging.getLogger(__name__)

router = APIRouter()


@router.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> ClassifyResponse:
    if not req.summary.strip():
        raise HTTPException(status_code=422, detail="summary must be non-empty")
    try:
        result = classifier_service.run(
            summary=req.summary,
            geo_cities=[c.model_dump() for c in req.geo_cities],
            search_tags=req.search_tags,
            source_profile=req.source_profile.model_dump() if req.source_profile else None,
            geo_scope=req.geo_scope,
        )
    except FileNotFoundError as exc:
        log.error("topics.yaml missing: %s", exc, extra={"article_id": req.article_id})
        raise HTTPException(status_code=503, detail="taxonomy_missing")
    mark_warm("classify")
    return ClassifyResponse(
        article_id=req.article_id,
        topics=result["topics"],
        scores=result["scores"],
        geo_scope=result["geo_scope"],
        out_of_scope=result.get("out_of_scope", False),
    )
