import pytest
from nlp.extractor.service import extract_and_embed, _textrank_extract, _split_sentences


def test_split_sentences_basic():
    text = "Primera frase. Segunda frase. Tercera frase."
    sentences = _split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "Primera frase."


def test_short_text_returned_unchanged():
    text = "Solo una frase."
    result = _textrank_extract(text, max_words=200)
    assert result == text


def test_two_sentence_text_returned_unchanged():
    text = "Primera frase. Segunda frase."
    result = _textrank_extract(text, max_words=200)
    assert result == text


def test_textrank_respects_max_words():
    sentences = [f"Esta es la frase número {i} sobre ciclismo en la ciudad." for i in range(10)]
    text = " ".join(sentences)
    result = _textrank_extract(text, max_words=30)
    word_count = len(result.split())
    assert word_count <= 40  # allow slight overshoot from sentence boundaries


def test_textrank_returns_non_empty_for_long_text(madrid_text):
    result = _textrank_extract(madrid_text, max_words=200)
    assert len(result.strip()) > 0


def test_extract_and_embed_returns_correct_shapes(madrid_text):
    extract, embedding = extract_and_embed(madrid_text)
    assert isinstance(extract, str)
    assert len(extract) > 0
    assert embedding.shape == (384,)
    assert embedding.dtype.name == "float32"
