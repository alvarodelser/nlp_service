# nlp_service/nlp/dedup/minhash.py
"""Stateless MinHash primitive. The orchestrator stores compute(text) as a Weaviate property and
uses jaccard() to catch easy duplicates. No in-process index."""
import os
import unicodedata

from datasketch import MinHash

_NUM_PERM = int(os.environ.get("MINHASH_NUM_PERM", "128"))
_SHINGLE_SIZE = 3


def _normalize(text: str) -> str:
    text = text.lower()
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")


def _shingles(text: str) -> list[str]:
    words = _normalize(text).split()
    if len(words) < _SHINGLE_SIZE:
        return [" ".join(words)] if words else []
    return [" ".join(words[i:i + _SHINGLE_SIZE]) for i in range(len(words) - _SHINGLE_SIZE + 1)]


def compute(text: str) -> list[int]:
    """MinHash signature as a plain int list — storable as a Weaviate property."""
    mh = MinHash(num_perm=_NUM_PERM)
    for s in _shingles(text):
        mh.update(s.encode("utf-8"))
    return mh.hashvalues.tolist()


def jaccard(a: list[int], b: list[int]) -> float:
    """Estimated Jaccard from two equal-length signatures."""
    if not a or len(a) != len(b):
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)
