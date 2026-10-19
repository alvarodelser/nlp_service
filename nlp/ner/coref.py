# nlp_service/nlp/ner/coref.py
"""Coreference, two ways.

resolve_pronouns: document-level pronoun rewriting. The orchestrator runs it BEFORE chunking,
since antecedents frequently sit in an earlier chunk. Per-chunk extract() never calls it.

link_references: span-based linking that leaves the text untouched. Given a passage and the
entities already known for its document set, it returns each expression that refers to one of
them without naming it ("la empresa", "la adjudicataria", "dicha entidad", pronouns), quoted
literally so the caller can locate it and keep its offsets (provenance)."""
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


_LINKS_PROMPT_PATH = Path(__file__).parent / "prompts" / "coref_links.txt"
_LINKS_TEMPLATE: str | None = None


def _links_template() -> str:
    global _LINKS_TEMPLATE
    if _LINKS_TEMPLATE is None:
        _LINKS_TEMPLATE = _LINKS_PROMPT_PATH.read_text(encoding="utf-8")
    return _LINKS_TEMPLATE


def _links_grammar(n_candidates: int) -> dict:
    return {
        "type": "object", "required": ["links"],
        "properties": {"links": {"type": "array", "items": {
            "type": "object", "required": ["mention", "refers_to", "evidence"],
            "properties": {"mention": {"type": "string"},
                           "refers_to": {"type": "integer", "minimum": 1, "maximum": max(1, n_candidates)},
                           "evidence": {"type": "string"}}}}},
    }


def link_references(text: str, candidates: list[dict], context: str | None = None) -> list[dict]:
    """candidates: [{name, type, description?}]. Returns [{mention, evidence, candidate (0-based)}].
    Raises httpx.HTTPError when Ollama is unavailable (the caller decides what best-effort means)."""
    if not text.strip() or not candidates:
        return []
    listing = "\n".join(f"{i}. [{c.get('type', '')}] {c['name']}" + (f" — {c['description'][:160]}" if c.get("description") else "")
                         for i, c in enumerate(candidates, 1))
    user = (f"{('CONTEXTO: ' + context + chr(10) + chr(10)) if context else ''}ENTIDADES CONOCIDAS\n{listing}\n\n"
            f"FRAGMENTO\n{text}")
    out = ollama_client.extract(_links_template(), user, _links_grammar(len(candidates)))
    links = []
    for link in out.get("links", []):
        k = link.get("refers_to")
        mention = (link.get("mention") or "").strip()
        if isinstance(k, int) and 1 <= k <= len(candidates) and mention:
            links.append({"mention": mention, "evidence": (link.get("evidence") or "").strip(), "candidate": k - 1})
    return links
