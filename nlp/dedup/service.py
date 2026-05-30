import numpy as np
import faiss
from datasketch import MinHashLSH

from nlp.dedup.persistence import (
    load_minhash, save_minhash, load_faiss, save_faiss,
    text_to_minhash, EMBED_DIM,
)

COSINE_THRESHOLD = 0.92

_lsh: MinHashLSH | None = None
_faiss_index: faiss.IndexFlatIP | None = None
_faiss_ids: list[str] = []


def startup() -> None:
    global _lsh, _faiss_index, _faiss_ids
    _lsh = load_minhash()
    _faiss_index, _faiss_ids = load_faiss()


def check_minhash(article_id: str, text: str) -> str | None:
    m = text_to_minhash(text)
    results = _lsh.query(m)
    if results:
        return results[0]
    _lsh.insert(article_id, m)
    save_minhash(_lsh)
    return None


def check_embedding(article_id: str, embedding: np.ndarray) -> str | None:
    vec = embedding.reshape(1, -1).astype(np.float32)
    if _faiss_index.ntotal > 0:
        distances, indices = _faiss_index.search(vec, 1)
        if distances[0][0] >= COSINE_THRESHOLD:
            return _faiss_ids[indices[0][0]]

    _faiss_index.add(vec)
    _faiss_ids.append(article_id)
    save_faiss(_faiss_index, _faiss_ids)
    return None
