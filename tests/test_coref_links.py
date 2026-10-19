"""Span-based reference linking: the prompt gets the numbered candidates, the answer is mapped
back to 0-based candidates and anything out of range or empty is dropped."""
from nlp.ner import coref


def test_link_references_maps_and_filters(monkeypatch):
    seen = {}

    def fake_extract(system, user, grammar):
        seen["user"], seen["grammar"] = user, grammar
        return {"links": [{"mention": "la adjudicataria", "refers_to": 2, "evidence": "la adjudicataria presentó"},
                          {"mention": "", "refers_to": 1, "evidence": "x"},
                          {"mention": "ella", "refers_to": 7, "evidence": "ella"}]}

    monkeypatch.setattr(coref.ollama_client, "extract", fake_extract)
    cands = [{"name": "Ayuntamiento de Benabarre", "type": "CONTRACTING_AUTHORITY"},
             {"name": "ESPUBLICO SERVICIOS PARA LA ADMINISTRACIÓN S.A.", "type": "COMPANY"}]
    out = coref.link_references("Resuelve adjudicar... la adjudicataria presentó la garantía.", cands, "Resolución de adjudicación")
    assert out == [{"mention": "la adjudicataria", "evidence": "la adjudicataria presentó", "candidate": 1}]
    assert "2. [COMPANY] ESPUBLICO" in seen["user"] and "CONTEXTO: Resolución" in seen["user"]
    assert seen["grammar"]["properties"]["links"]["items"]["properties"]["refers_to"]["maximum"] == 2


def test_no_candidates_no_call(monkeypatch):
    monkeypatch.setattr(coref.ollama_client, "extract", lambda *a: (_ for _ in ()).throw(AssertionError("called")))
    assert coref.link_references("texto", []) == []
