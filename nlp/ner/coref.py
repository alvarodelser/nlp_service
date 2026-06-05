# nlp_service/nlp/ner/coref.py
"""Document-level pronoun resolution. The orchestrator runs this BEFORE chunking, since
antecedents frequently sit in an earlier chunk. Per-chunk extract() never calls it."""
import logging
from pathlib import Path

import httpx

from . import ollama_client

log = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "coref.txt"
_TEMPLATE: str | None = None


def _template() -> str:
    global _TEMPLATE
    if _TEMPLATE is None:
        _TEMPLATE = _PROMPT_PATH.read_text(encoding="utf-8")
    return _TEMPLATE


def resolve_pronouns(document_text: str) -> str:
    """Rewrite third-person pronouns to their antecedent's name; otherwise return the text
    unchanged. On LLM failure, returns the original text (resolution is best-effort)."""
    if not document_text.strip():
        return document_text
    try:
        return ollama_client.rewrite(system=_template(), user=document_text)
    except (httpx.HTTPError, KeyError) as exc:
        log.warning("coref failed; returning original text: %s", exc)
        return document_text
