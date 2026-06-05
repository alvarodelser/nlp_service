# nlp_service/api/main.py
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response

from api.warmth import get_missing

logging.basicConfig(
    level=os.environ.get("NLP_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("nlp_service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("nlp-service starting up")

    import torch
    import flair as _flair_lib

    # Set Flair device before any model loads
    _flair_lib.device = torch.device(os.environ.get("NER_DEVICE", "cpu"))

    from nlp import nli as _nli
    from nlp.encoder import load_encoder as _load_encoder
    from nlp.geotagger import ner as _ner
    from nlp.geotagger import service as _geo_svc
    from nlp.dedup import service as _dedup_svc
    from nlp.dedup import embedding_index as _emb_idx
    from api.warmth import mark_warm as _mark_warm

    _load_encoder()
    _nli._ensure_loaded()       # shared by /nli + geotagger
    _ner._ensure_loaded()
    _geo_svc.load()
    _dedup_svc.load()
    _emb_idx._ensure_loaded()

    _mark_warm("geotag")
    _mark_warm("nli")
    _mark_warm("dedup")
    _mark_warm("summarize")
    log.info("nlp-service ready")
    yield
    log.info("nlp-service shutting down — flushing dedup state")
    try:
        from nlp.dedup import service as dedup_service
        dedup_service.flush()
    except Exception:
        log.exception("dedup flush on shutdown failed (continuing)")


app = FastAPI(title="NLP Service", lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/readyz")
def readyz(response: Response) -> dict:
    expected = {"summarize", "geotag", "nli", "dedup"}
    missing = get_missing(expected)
    if missing:
        response.status_code = 503
        return {"status": "warming", "missing": missing}
    return {"status": "ready"}


def _register_routers() -> None:
    from api.routers import summarize as summarize_router
    from api.routers import geotag as geotag_router
    from api.routers import nli as nli_router
    from api.routers import dedup as dedup_router
    from api.routers import ollama as ollama_router
    app.include_router(summarize_router.router)
    app.include_router(geotag_router.router)
    app.include_router(nli_router.router)
    app.include_router(dedup_router.router)
    app.include_router(ollama_router.router)


_register_routers()
