from unittest.mock import MagicMock, patch
import pytest


def _mock_taxonomy():
    tax = MagicMock()
    tax.relevance_hypothesis = ""
    tax.relevance_threshold = 0.4
    tax.blacklist_labels = []
    tax.blacklist_threshold = 0.7
    tax.labels = ["carril bici"]
    tax.score_threshold = 0.5
    tax.top_k = 3
    tax.scope_hypotheses = {"city": "hip city", "national": "hip national"}
    tax.scope_threshold = 0.35
    return tax


def test_geo_scope_provided_skips_scope_nli(madrid_text):
    topic_result = {"labels": ["carril bici"], "scores": [0.8], "sequence": ""}
    with patch("nlp.classifier.service.taxonomy") as mock_tax, \
         patch("nlp.classifier.model.classify") as mock_classify:
        mock_tax.load.return_value = _mock_taxonomy()
        mock_classify.return_value = topic_result

        from nlp.classifier.service import run
        result = run(
            summary=madrid_text,
            geo_cities=[],
            geo_scope="city",
        )

    assert mock_classify.call_count == 1
    assert result["geo_scope"] == "city"


def test_geo_scope_absent_runs_scope_nli(madrid_text):
    topic_result = {"labels": ["carril bici"], "scores": [0.8], "sequence": ""}
    # scope_pass calls classify once per scope hypothesis (2 in mock taxonomy)
    scope_city_result = {"labels": ["hip city"], "scores": [0.9], "sequence": ""}
    scope_national_result = {"labels": ["hip national"], "scores": [0.3], "sequence": ""}
    call_results = [topic_result, scope_city_result, scope_national_result]

    with patch("nlp.classifier.service.taxonomy") as mock_tax, \
         patch("nlp.classifier.model.classify", side_effect=call_results) as mock_classify:
        mock_tax.load.return_value = _mock_taxonomy()

        from nlp.classifier.service import run
        result = run(summary=madrid_text, geo_cities=[], geo_scope=None)

    assert mock_classify.call_count == 3
