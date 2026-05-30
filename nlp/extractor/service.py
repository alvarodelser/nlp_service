import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nlp.encoder import encode


def extract_and_embed(text: str, max_words: int = 200) -> tuple[str, np.ndarray]:
    extract = _tfidf_extract(text, max_words)
    return extract, encode(extract)


def _tfidf_extract(text: str, max_words: int) -> str:
    sentences = _split_sentences(text)
    if len(sentences) <= 2:
        return text

    try:
        tfidf = TfidfVectorizer().fit_transform(sentences)
    except ValueError:
        return " ".join(sentences[:3])

    scores = np.asarray(tfidf.sum(axis=1)).ravel()
    ranked = sorted(zip(scores, sentences), reverse=True)

    result, word_count = [], 0
    for _, sent in ranked:
        wc = len(sent.split())
        if word_count + wc > max_words:
            break
        result.append(sent)
        word_count += wc

    return " ".join(result) if result else sentences[0]


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
