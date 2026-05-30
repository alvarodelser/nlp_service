from fastapi import APIRouter
from api.models import ClassifyRequest, ClassifyResponse
from nlp.classifier.service import classify

router = APIRouter()


@router.post("/classify", response_model=ClassifyResponse)
def classify_article(req: ClassifyRequest) -> ClassifyResponse:
    return classify(
        summary=req.summary,
        geo_cities=req.geo_cities,
        search_tags=req.search_tags,
        source_profile=req.source_profile,
    )
