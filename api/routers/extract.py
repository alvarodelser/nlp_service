from fastapi import APIRouter
from api.models import ExtractRequest, ExtractResponse
from nlp.extractor.service import extract_and_embed

router = APIRouter()


@router.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest) -> ExtractResponse:
    extract_text, embedding = extract_and_embed(req.text)
    return ExtractResponse(extract=extract_text, embedding_raw=embedding.tolist())
