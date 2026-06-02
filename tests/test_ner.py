from unittest.mock import MagicMock, patch
import pytest
from nlp.geotagger.ner import Span


def _make_flair_entity(text, tag, start, end):
    e = MagicMock()
    e.text = text
    e.tag = tag
    e.start_position = start
    e.end_position = end
    return e


def _make_mock_tagger(entities):
    tagger = MagicMock()

    def predict_side_effect(sentence):
        sentence.get_spans.return_value = entities

    tagger.predict.side_effect = predict_side_effect
    return tagger


def _make_mock_flair_sentence():
    def sentence_constructor(text, use_tokenizer=True):
        sentence = MagicMock()
        sentence.get_spans.return_value = []
        return sentence
    return sentence_constructor


def test_extract_spans_returns_only_loc(madrid_text):
    entities = [
        _make_flair_entity("Madrid", "LOC", 19, 25),
        _make_flair_entity("Juan García", "PER", 50, 61),
    ]
    mock_tagger = _make_mock_tagger(entities)

    with patch("nlp.geotagger.ner._tagger", mock_tagger), \
         patch("nlp.geotagger.ner.FlairSentence", _make_mock_flair_sentence()):
        from nlp.geotagger.ner import extract_spans
        spans = extract_spans(madrid_text)

    assert len(spans) == 1
    assert spans[0].text == "Madrid"
    assert spans[0].label == "LOC"


def test_extract_spans_includes_street_regex_hits():
    text = "El corte afecta a la Calle Alcalá entre Goya y Velázquez."
    entities = []  # Flair finds nothing
    mock_tagger = _make_mock_tagger(entities)

    with patch("nlp.geotagger.ner._tagger", mock_tagger), \
         patch("nlp.geotagger.ner.FlairSentence", _make_mock_flair_sentence()):
        from nlp.geotagger.ner import extract_spans
        spans = extract_spans(text)

    street_spans = [s for s in spans if s.hint == "street"]
    assert len(street_spans) >= 1
    assert any("Calle Alcalá" in s.text for s in street_spans)


def test_extract_spans_no_subword_artifacts():
    entities = [
        _make_flair_entity("Eixample", "LOC", 15, 23),
    ]
    mock_tagger = _make_mock_tagger(entities)

    with patch("nlp.geotagger.ner._tagger", mock_tagger), \
         patch("nlp.geotagger.ner.FlairSentence", _make_mock_flair_sentence()):
        from nlp.geotagger.ner import extract_spans
        spans = extract_spans("Los vecinos del Eixample se quejan.")

    assert all("##" not in s.text for s in spans)
    assert spans[0].text == "Eixample"


def test_span_dataclass_fields():
    span = Span(text="Madrid", label="LOC", start_char=0, end_char=6)
    assert span.text == "Madrid"
    assert span.label == "LOC"
    assert span.hint == ""
