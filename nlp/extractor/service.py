import re
import numpy as np
from nlp.encoder import encode
from nlp.extractor.extractive import extract_top_sentences


def extract_and_embed(text: str, max_words: int = 200) -> tuple[str, np.ndarray]:
    extract = _textrank_extract(text, max_words)
    return extract, encode(extract)


def _textrank_extract(text: str, max_words: int) -> str:
    sentences = _split_sentences(text)
    if len(sentences) <= 2:
        return text
    avg_words = sum(len(s.split()) for s in sentences) / len(sentences)
    n = max(1, int(max_words / max(avg_words, 1)))
    return extract_top_sentences(text, n)


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
