from unittest.mock import MagicMock, patch
import pytest


def _nli_scope_result(winner: str):
    """Return a mock nli.classify result where winner scores 0.9."""
    h_local = (
        "Este artículo describe actuaciones, obras o iniciativas "
        "de un ayuntamiento o municipio concreto de España."
    )
    h_regional = (
        "Este artículo describe políticas o actuaciones de una comunidad autónoma, "
        "diputación provincial o región de España que afectan a varios municipios."
    )
    h_national = (
        "Este artículo describe una política, ley, normativa o acontecimiento "
        "de alcance estatal en España, sin limitarse a una ciudad o región concreta."
    )
    scores = {h_local: 0.1, h_regional: 0.1, h_national: 0.1}
    winner_map = {"local": h_local, "regional": h_regional, "national": h_national}
    scores[winner_map[winner]] = 0.9
    labels = sorted(scores, key=lambda k: scores[k], reverse=True)
    return {"labels": labels, "scores": [scores[l] for l in labels], "sequence": ""}


def _nli_city_result(winner_city: str, all_cities: list[str]):
    scores = {c: 0.1 for c in all_cities}
    scores[winner_city] = 0.9
    labels = sorted(scores, key=lambda k: scores[k], reverse=True)
    return {"labels": labels, "scores": [scores[l] for l in labels], "sequence": ""}


def test_scope_national_returns_no_city(national_text):
    with patch("nlp.geotagger.service.ner") as mock_ner, \
         patch("nlp.geotagger.service.gazetteer") as mock_geo, \
         patch("nlp.nli.classify") as mock_nli:

        mock_ner.extract_spans.return_value = []
        mock_geo.lookup.return_value = []
        mock_geo.lookup_street.return_value = []
        mock_geo.lookup_street_all_cities.return_value = {}
        mock_geo.get_city_prior.return_value = None
        mock_geo._STREET_PREFIX_RE.match.return_value = None
        mock_nli.return_value = _nli_scope_result("national")

        from nlp.geotagger import service
        service._cities = []
        service._by_name = {}
        result = service.run(national_text, headline="", source="")

    assert result["geo_scope"] == "national"
    assert result["geo_cities"] == []


def test_scope_local_single_city_no_stage2(madrid_text):
    from nlp.geotagger.gazetteer import GeoEntry
    from nlp.geotagger.ner import Span

    madrid_entry = GeoEntry(
        geonames_id=3117735, name="Madrid", lat=40.4165, lon=-3.7026,
        feature_class="P", feature_code="PPLC", admin1_code="29", population=3200000,
    )
    madrid_span = Span(text="Madrid", label="LOC", start_char=19, end_char=25)

    with patch("nlp.geotagger.service.ner") as mock_ner, \
         patch("nlp.geotagger.service.gazetteer") as mock_geo, \
         patch("nlp.nli.classify") as mock_nli:

        mock_ner.extract_spans.return_value = [madrid_span]
        mock_geo.lookup.side_effect = lambda span: [madrid_entry] if "Madrid" in span else []
        mock_geo.lookup_street.return_value = []
        mock_geo.lookup_street_all_cities.return_value = {}
        mock_geo.get_city_prior.return_value = None
        mock_geo._STREET_PREFIX_RE.match.return_value = None
        mock_nli.return_value = _nli_scope_result("local")

        from nlp.geotagger import service
        service._cities = [{"id": 3117735, "name": "Madrid", "population": 3200000}]
        service._by_name = {"madrid": {"id": 3117735, "name": "Madrid", "population": 3200000}}
        result = service.run(madrid_text, headline="Carril bici en Madrid", source="")

    assert result["geo_scope"] == "city"
    assert result["geo_cities"][0]["city_name"] == "Madrid"
    # Stage 2 NLI must NOT fire for a single-candidate city pool
    assert mock_nli.call_count == 1


def test_street_rescue_routes_to_geo_streets():
    from nlp.geotagger.ner import Span

    calle_span = Span(text="Calle Alcalá", label="LOC", start_char=10, end_char=22)

    with patch("nlp.geotagger.service.ner") as mock_ner, \
         patch("nlp.geotagger.service.gazetteer") as mock_geo, \
         patch("nlp.nli.classify") as mock_nli:

        mock_ner.extract_spans.return_value = [calle_span]
        mock_geo.lookup.return_value = []
        import re
        mock_geo._STREET_PREFIX_RE = re.compile(
            r'^(?:calle|avda?\.?|avenida|plaza)\s+', re.IGNORECASE
        )
        mock_geo.lookup_street.return_value = [101, 102]
        mock_geo.lookup_street_all_cities.return_value = {}
        mock_geo.get_city_prior.return_value = None
        mock_nli.return_value = _nli_scope_result("local")

        from nlp.geotagger import service
        service._cities = []
        service._by_name = {}
        result = service.run(
            "El corte afecta a la Calle Alcalá.",
            headline="",
            source="",
        )

    street_places = [p for p in result["all_places"] if p["type"] == "street"]
    assert len(street_places) >= 1
    assert any("Alcalá" in p["text"] for p in street_places)
