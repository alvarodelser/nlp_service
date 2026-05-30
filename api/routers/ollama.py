# nlp_service/api/routers/ollama.py
"""
Thin proxy to the Ollama sidecar.  Lets notebooks running locally call Ollama
through the NLP service (which is port-forwarded / nginx-proxied) without
needing a direct connection to the Docker network.

POST /ollama/generate  →  Ollama /api/generate  (returns raw JSON)
GET  /ollama/tags      →  Ollama /api/tags       (model list)
"""
import logging
import os

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

log = logging.getLogger("nlp_service.ollama_proxy")
router = APIRouter(prefix="/ollama")

_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "120"))


async def _forward(method: str, path: str, body: bytes | None = None) -> JSONResponse:
    url = f"{_OLLAMA_HOST}{path}"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.request(method, url, content=body,
                                        headers={"Content-Type": "application/json"})
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except httpx.ConnectError as exc:
        log.error("ollama unreachable: %s", exc)
        raise HTTPException(status_code=503, detail="ollama_unreachable")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="ollama_timeout")


@router.post("/generate")
async def generate(request: Request) -> JSONResponse:
    body = await request.body()
    return await _forward("POST", "/api/generate", body)


@router.get("/tags")
async def tags() -> JSONResponse:
    return await _forward("GET", "/api/tags")
