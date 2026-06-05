# nlp_service/nlp/dedup/llm.py
"""Generic LLM 'same thing?' adjudication. Compares two texts; `kind` only flavours the prompt,
the logic is identical. Returns 'yes' | 'no' | 'unsure'; on any error returns 'unsure'."""
import json
import logging
import os

import httpx

log = logging.getLogger(__name__)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
ADJ_MODEL   = os.environ.get("EXTRACTION_MODEL", "qwen2.5:32b")
ADJ_TIMEOUT = float(os.environ.get("DEDUP_LLM_TIMEOUT", "30"))

_SCHEMA = {
    "type": "object",
    "required": ["verdict"],
    "properties": {"verdict": {"type": "string", "enum": ["yes", "no", "unsure"]}},
}
_QUESTION = {
    "article":  "¿Estos dos textos se refieren a lo mismo?",
    "entity":   "¿Estas dos descripciones se refieren a la misma entidad real?",
    "relation": "¿Estas dos descripciones se refieren a la misma relación?",
}


def adjudicate(kind: str, text_a: str, text_b: str) -> str:
    question = _QUESTION.get(kind, "¿Estos dos elementos son el mismo?")
    prompt = (f"{question}\n\nA:\n{text_a}\n\nB:\n{text_b}\n\n"
              "Responde 'yes', 'no' o 'unsure'.")
    try:
        r = httpx.post(
            f"{OLLAMA_HOST}/api/chat",
            json={"model": ADJ_MODEL,
                  "messages": [{"role": "user", "content": prompt}],
                  "stream": False, "format": _SCHEMA, "options": {"temperature": 0}},
            timeout=ADJ_TIMEOUT,
        )
        r.raise_for_status()
        return json.loads(r.json()["message"]["content"])["verdict"]
    except (httpx.HTTPError, KeyError, json.JSONDecodeError, ValueError) as exc:
        log.warning("dedup adjudication failed -> unsure: %s", exc)
        return "unsure"
