from pipelines.relations import chunker


def test_empty_text_returns_no_chunks():
    assert chunker.split("") == []
    assert chunker.split("   \n  ") == []


def test_short_text_is_single_chunk():
    out = chunker.split("hola mundo esto es corto", max_words=512)
    assert out == ["hola mundo esto es corto"]


def test_paragraphs_under_budget_combine_into_one_chunk():
    text = "uno dos tres\n\ncuatro cinco seis"
    out = chunker.split(text, max_words=10)
    assert len(out) == 1
    assert "uno dos tres" in out[0] and "cuatro cinco seis" in out[0]


def test_paragraphs_flush_when_budget_exceeded():
    # each paragraph 3 words, budget 4 → each paragraph in its own chunk
    text = "uno dos tres\n\ncuatro cinco seis\n\nsiete ocho nueve"
    out = chunker.split(text, max_words=4)
    assert len(out) == 3


def test_single_paragraph_longer_than_budget_is_hard_split():
    text = " ".join(f"w{i}" for i in range(10))    # 10 words, budget 4
    out = chunker.split(text, max_words=4)
    assert len(out) == 3                            # 4 + 4 + 2
    assert all(len(c.split()) <= 4 for c in out)
    assert " ".join(out).split() == text.split()    # no words lost


def test_env_default_used_when_max_words_none(monkeypatch):
    monkeypatch.setenv("RELATIONS_CHUNK_TOKENS", "3")
    text = "a b c d e f"
    out = chunker.split(text)
    assert all(len(c.split()) <= 3 for c in out)
    assert len(out) == 2
