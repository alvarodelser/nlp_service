# nlp_service/nlp/summarizer/service.py
import logging
import os
from pathlib import Path

from . import ollama_client, profiles
from .extractive import extract_top_sentences

log = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).parent / "prompts"
_TEMPLATE_CACHE: dict[str, str] = {}

# Single budget for the extractive pre-step. Its only purpose is to keep the prompt input within
# the LLM's context window (the same model for every profile), so there is one value, not one per
# profile. Words are a practical proxy for tokens.
_MAX_INPUT_WORDS = int(os.environ.get("SUMMARIZE_MAX_INPUT_WORDS", "400"))


def load() -> None:
    """Warm the article template (kept for the api warmup hook)."""
    _template("article.es.txt")


def _template(prompt_file: str) -> str:
    if prompt_file not in _TEMPLATE_CACHE:
        _TEMPLATE_CACHE[prompt_file] = (_PROMPT_DIR / prompt_file).read_text(encoding="utf-8")
    return _TEMPLATE_CACHE[prompt_file]


def _reduce(text: str) -> str:
    """Extractive (TextRank) reduction, only when over the context-window budget."""
    if len(text.split()) <= _MAX_INPUT_WORDS:
        return text
    sentences = [s for s in text.replace("\n", " ").split(". ") if s.strip()]
    if len(sentences) <= 2:
        return text
    avg_words = sum(len(s.split()) for s in sentences) / len(sentences)
    n = max(1, int(_MAX_INPUT_WORDS / max(avg_words, 1)))
    return extract_top_sentences(text, n)


def _prepare_fields(profile: profiles.Profile, fields: dict) -> dict:
    """Normalise fields for the prompt: join evidence/description lists, apply extractive
    reduction on the profile's designated long field."""
    prepared = dict(fields)
    if isinstance(prepared.get("evidence"), list):
        prepared["evidence_text"] = "\n".join(
            f"{i + 1}. {e}" for i, e in enumerate(prepared["evidence"])
        )
    if isinstance(prepared.get("descriptions"), list):
        prepared["descriptions_text"] = "\n".join(f"- {d}" for d in prepared["descriptions"])
    subtype = prepared.get("subtype")
    prepared["subtype_line"] = f"Subtipo: {subtype}" if subtype else ""
    if profile.extract_field and profile.extract_field in prepared:
        prepared[profile.extract_field] = _reduce(str(prepared[profile.extract_field]))
    return prepared


def summarize(profile_name: str, fields: dict) -> dict:
    """Generate text for the named profile. Returns the profile's output_keys.

    Raises ValueError for an unknown profile.
    Raises httpx.HTTPError if Ollama is unavailable, KeyError/ValueError on malformed JSON.
    """
    profile = profiles.get(profile_name)
    prepared = _prepare_fields(profile, fields)
    prompt = _template(profile.prompt_file).format(**prepared)

    result = ollama_client.generate(prompt, schema=profile.schema)

    if profile.validate is not None:
        ok, reason = profile.validate(result)
        if not ok:
            log.info("validator rejected (%s); retry once with tightened prompt", reason)
            result = ollama_client.generate(prompt + profile.retry_suffix, schema=profile.schema)

    return {k: result[k] for k in profile.output_keys}
