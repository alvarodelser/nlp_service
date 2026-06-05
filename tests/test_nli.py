from unittest.mock import MagicMock, patch
import pytest


def _make_mock_pipeline(labels, scores):
    mock = MagicMock()
    mock.return_value = {"labels": labels, "scores": scores, "sequence": "test"}
    return mock


def test_classify_returns_expected_keys():
    with patch("nlp.nli._pipeline", _make_mock_pipeline(["a", "b"], [0.8, 0.2])):
        from nlp import nli
        result = nli.classify("some text", labels=["a", "b"], multi_label=False)
    assert "labels" in result
    assert "scores" in result


def test_classify_passes_hypothesis_template():
    mock_pipe = _make_mock_pipeline(["Madrid", "Barcelona"], [0.7, 0.3])
    with patch("nlp.nli._pipeline", mock_pipe):
        from nlp import nli
        nli.classify(
            "El ayuntamiento trabaja en Madrid.",
            labels=["Madrid", "Barcelona"],
            multi_label=False,
            hypothesis_template="Este artículo trata sobre {}.",
        )
    call_kwargs = mock_pipe.call_args[1]
    assert call_kwargs["hypothesis_template"] == "Este artículo trata sobre {}."


def test_classify_multi_label_flag_passed():
    mock_pipe = _make_mock_pipeline(["topic"], [0.9])
    with patch("nlp.nli._pipeline", mock_pipe):
        from nlp import nli
        nli.classify("text", labels=["topic"], multi_label=True)
    assert mock_pipe.call_args[1]["multi_label"] is True


def test_score_short_circuits_on_first_below_threshold():
    mock_pipe = MagicMock(side_effect=lambda text, candidate_labels, multi_label,
                          hypothesis_template: {"labels": list(candidate_labels),
                          "scores": [{"a": 0.9, "b": 0.2, "c": 0.8}[candidate_labels[0]]],
                          "sequence": text})
    with patch("nlp.nli._pipeline", mock_pipe):
        from nlp import nli
        out = nli.score("t", ["a", "b", "c"], threshold=0.5)
    assert [d["hypothesis"] for d in out] == ["a", "b"]   # stops at b (0.2 < 0.5); c not run
