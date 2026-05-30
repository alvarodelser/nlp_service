from contextlib import asynccontextmanager
from fastapi import FastAPI

from nlp.encoder import load_encoder
from nlp.dedup import service as dedup_svc
from nlp.geotagger import service as geo_svc
from nlp.classifier import service as cls_svc

from api.routers import extract, dedup, summarize, geotag, classify, ollama


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_encoder()
    dedup_svc.startup()
    geo_svc.startup()
    cls_svc.startup()
    yield


app = FastAPI(title="NLP Service — Alternative Mobility News", lifespan=lifespan)

app.include_router(extract.router)
app.include_router(dedup.router)
app.include_router(summarize.router)
app.include_router(geotag.router)
app.include_router(classify.router)
app.include_router(ollama.router, prefix="/ollama")
