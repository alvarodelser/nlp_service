from fastapi import APIRouter
from api.models import SummarizeRequest, SummarizeResponse
from nlp.summarizer.service import summarize

router = APIRouter()


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_article(req: SummarizeRequest) -> SummarizeResponse:
    headline, summary, embedding = await summarize(req.text, req.extract, req.headline)
    return SummarizeResponse(
        headline=headline,
        summary=summary,
        embedding_summary=embedding.tolist(),
    )
