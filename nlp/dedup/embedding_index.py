# nlp_service/nlp/dedup/embedding_index.py
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_DIM = 384

_model: SentenceTransformer | None = None


def _ensure_loaded() -> None:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)


def encode(text: str) -> np.ndarray:
    _ensure_loaded()
    assert _model is not None
    vec = _model.encode([text], normalize_embeddings=True)[0]
    return vec.astype("float32")


class EmbeddingIndex:
    def __init__(self, threshold: float | None = None) -> None:
        self.threshold = threshold if threshold is not None else float(
            os.environ.get("DEDUP_EMBED_THRESHOLD", "0.85")
        )
        self._index = faiss.IndexFlatIP(_DIM)

    def query(self, text: str) -> tuple[int | None, float]:
        """Returns (best_row, cosine_score) or (None, 0.0)."""
        return self.query_vec(encode(text))

    def query_vec(self, vec: np.ndarray) -> tuple[int | None, float]:
        """Query with a pre-computed embedding (e.g. from /extract)."""
        if self._index.ntotal == 0:
            return None, 0.0
        v = vec.reshape(1, -1).astype("float32")
        scores, rows = self._index.search(v, k=1)
        best_row = int(rows[0][0])
        best_score = float(scores[0][0])
        if best_row < 0:
            return None, 0.0
        return best_row, best_score

    def add(self, text: str) -> int:
        """Returns the row index assigned to this article."""
        return self.add_vec(encode(text))

    def add_vec(self, vec: np.ndarray) -> int:
        """Add a pre-computed embedding. Returns the row index assigned."""
        v = vec.reshape(1, -1).astype("float32")
        row = self._index.ntotal
        self._index.add(v)
        return row

    def serialize(self) -> bytes:
        return faiss.serialize_index(self._index)

    def deserialize(self, data: bytes) -> None:
        arr = np.frombuffer(data, dtype=np.uint8)
        self._index = faiss.deserialize_index(arr)

    def __len__(self) -> int:
        return self._index.ntotal
