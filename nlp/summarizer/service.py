# nlp_service/nlp/summarizer/service.py
import logging
from pathlib import Path

from . import ollama_client, validator

log = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "rewrite.es.txt"
_PROMPT_TEMPLATE: str | None = None
_RETRY_SUFFIX = "\nRECUERDA: titular 8-15 palabras, resumen 2-4 frases."


def load() -> None:
    global _PROMPT_TEMPLATE
    if _PROMPT_TEMPLATE is None:
        _PROMPT_TEMPLATE = _PROMPT_PATH.read_text(encoding="utf-8")


def run(text: str, extract: str, headline: str) -> dict:
    """Returns {'headline': str, 'summary': str}.

    extract is pre-computed by /extract; text is retained but the prompt uses
    extract to stay within the LLM's context window.
    """
    load()
    assert _PROMPT_TEMPLATE is not None

    prompt = _PROMPT_TEMPLATE.format(headline=headline, extract=extract or text)
    result = ollama_client.generate(prompt)

    ok, reason = validator.validate(result)
    if not ok:
        log.info("validator rejected (%s), retrying once with tightened prompt", reason)
        result = ollama_client.generate(prompt + _RETRY_SUFFIX)

    return {"headline": result["headline"], "summary": result["summary"]}
