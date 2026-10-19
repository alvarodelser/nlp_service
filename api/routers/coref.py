# nlp_service/api/routers/coref.py
import logging

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from nlp.ner import coref

log = logging.getLogger(__name__)

router = APIRouter()


class CorefRequest(BaseModel):
    text: str


class Candidate(BaseModel):
    name: str
    type: str = ""
    description: str | None = None


class CorefLinksRequest(BaseModel):
    text: str
    candidates: list[Candidate]
    context: str | None = None      # e.g. the document's role and title


@router.post("/coref")
def rewrite(req: CorefRequest) -> dict:
    """Pronouns rewritten to their antecedents (best effort: the original text on failure)."""
    return {"text": coref.resolve_pronouns(req.text)}


@router.post("/coref/links")
def links(req: CorefLinksRequest) -> dict:
    """Expressions in `text` that refer to one of `candidates` without naming it, quoted literally."""
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must be non-empty")
    try:
        return {"links": coref.link_references(req.text, [c.model_dump() for c in req.candidates], req.context)}
    except httpx.HTTPError as exc:
        log.error("ollama unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="ollama_unavailable")
