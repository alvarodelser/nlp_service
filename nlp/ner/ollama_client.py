# nlp_service/nlp/ner/ollama_client.py
import json
import logging
import os
import time
from typing import Any

import httpx

log = logging.getLogger(__name__)

OLLAMA_HOST        = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
EXTRACTION_MODEL   = os.environ.get("EXTRACTION_MODEL", "qwen2.5:32b")
EXTRACTION_TIMEOUT = float(os.environ.get("EXTRACTION_TIMEOUT", "180"))
EXTRACTION_NUM_CTX = int(os.environ.get("EXTRACTION_NUM_CTX", "8192"))


def extract(system: str, user: str, grammar: dict, max_retries: int = 3) -> dict[str, Any]:
    """Constrained-decoding /api/chat call. Returns the parsed JSON object.

    `num_ctx` must hold prompt + text or Ollama silently truncates — size it to the schema.
    """
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = httpx.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model": EXTRACTION_MODEL,
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": user}],
                    "stream": False,
                    "format": grammar,
                    "options": {"temperature": 0, "num_ctx": EXTRACTION_NUM_CTX},
                },
                timeout=EXTRACTION_TIMEOUT,
            )
            response.raise_for_status()
            return json.loads(response.json()["message"]["content"])
        except (httpx.HTTPError, json.JSONDecodeError, KeyError) as exc:
            last_error = exc
            log.warning("ner extract failed (attempt %d/%d): %s", attempt + 1, max_retries, exc)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    assert last_error is not None
    raise last_error


def rewrite(system: str, user: str) -> str:
    """Plain-text /api/chat call (no JSON constraint) — used by coref."""
    response = httpx.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": EXTRACTION_MODEL,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": 0, "num_ctx": EXTRACTION_NUM_CTX},
        },
        timeout=EXTRACTION_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]
