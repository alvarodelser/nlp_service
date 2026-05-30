from fastapi import APIRouter
from api.models import GeotagRequest, GeotagResponse
from nlp.geotagger.service import geotag

router = APIRouter()


@router.post("/geotag", response_model=GeotagResponse)
def geotag_article(req: GeotagRequest) -> GeotagResponse:
    return geotag(req.text, req.headline)
