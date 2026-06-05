# nlp_service/nlp/resolver/ollama_client.py
import json
import os
from pathlib import Path

import httpx

OLLAMA_HOST     = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
RESOLVE_MODEL   = os.environ.get("RESOLVE_MODEL", "qwen2.5:32b")
RESOLVE_TIMEOUT = float(os.environ.get("RESOLVE_TIMEOUT", "180"))

_PROMPT_PATH = Path(__file__).parent / "prompts" / "resolution.txt"
_TEMPLATE: str | None = None

_CLUSTER_SCHEMA = {
    "type": "object",
    "required": ["clusters"],
    "properties": {
        "clusters": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["canonical_name", "subtype", "candidate_indices"],
                "properties": {
                    "canonical_name":    {"type": "string"},
                    "subtype":           {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "candidate_indices": {"type": "array", "items": {"type": "integer"}},
                },
            },
        },
    },
}


def _template() -> str:
    global _TEMPLATE
    if _TEMPLATE is None:
        _TEMPLATE = _PROMPT_PATH.read_text(encoding="utf-8")
    return _TEMPLATE


def refine(candidates) -> list[dict]:
    """Send pre-merged candidates (same type) to the LLM. Returns the clusters list, each with
    local candidate_indices. Raises httpx.HTTPError / ValueError on failure."""
    lines = []
    for i, c in enumerate(candidates):
        ev = c.evidence[0] if c.evidence else ""
        lines.append(f"{i}. nombre='{c.canonical_name}' tipo={c.type} evidencia='{ev}'")
    prompt = _template().format(candidate_list="\n".join(lines))
    response = httpx.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": RESOLVE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": _CLUSTER_SCHEMA,
            "options": {"temperature": 0},
        },
        timeout=RESOLVE_TIMEOUT,
    )
    response.raise_for_status()
    return json.loads(response.json()["message"]["content"])["clusters"]
