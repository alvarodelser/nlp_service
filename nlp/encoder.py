import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 384

_model: SentenceTransformer | None = None


def load_encoder() -> None:
    global _model
    _model = SentenceTransformer(MODEL_NAME)


def get_encoder() -> SentenceTransformer:
    if _model is None:
        load_encoder()
    return _model


def encode(text: str) -> np.ndarray:
    return get_encoder().encode(text, normalize_embeddings=True).astype(np.float32)
