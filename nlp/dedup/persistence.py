import os
import pickle
import faiss
import numpy as np
from datasketch import MinHashLSH, MinHash

MINHASH_PATH = os.getenv("MINHASH_INDEX_PATH", "data/minhash.pkl")
FAISS_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss.index")
EMBED_DIM = 384
NUM_PERM = 128


def load_minhash() -> MinHashLSH:
    if os.path.exists(MINHASH_PATH):
        with open(MINHASH_PATH, "rb") as f:
            return pickle.load(f)
    return MinHashLSH(threshold=0.75, num_perm=NUM_PERM)


def save_minhash(lsh: MinHashLSH) -> None:
    os.makedirs(os.path.dirname(MINHASH_PATH) or ".", exist_ok=True)
    with open(MINHASH_PATH, "wb") as f:
        pickle.dump(lsh, f)


def load_faiss() -> tuple[faiss.IndexFlatIP, list[str]]:
    if os.path.exists(FAISS_PATH):
        index = faiss.read_index(FAISS_PATH)
        id_path = FAISS_PATH + ".ids"
        with open(id_path, "rb") as f:
            ids = pickle.load(f)
        return index, ids
    return faiss.IndexFlatIP(EMBED_DIM), []


def save_faiss(index: faiss.IndexFlatIP, ids: list[str]) -> None:
    os.makedirs(os.path.dirname(FAISS_PATH) or ".", exist_ok=True)
    faiss.write_index(index, FAISS_PATH)
    with open(FAISS_PATH + ".ids", "wb") as f:
        pickle.dump(ids, f)


def text_to_minhash(text: str) -> MinHash:
    m = MinHash(num_perm=NUM_PERM)
    for token in _shingle(text, n=3):
        m.update(token.encode("utf-8"))
    return m


def _shingle(text: str, n: int) -> list[str]:
    tokens = text.lower().split()
    return [" ".join(tokens[i : i + n]) for i in range(max(1, len(tokens) - n + 1))]
