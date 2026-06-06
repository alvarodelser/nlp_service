import json
from unittest.mock import MagicMock

from nlp.ner.schema_types import ExtractedEntity, ExtractedRelation, NerResponse
from nlp.resolver.types import ResolvedEntity, ResolvedRelation, ResolveResponse
from pipelines.relations import orchestrator as orch

USE_CASE = "financial_flows"


def _ner_response():
    return NerResponse(
        entities=[
            ExtractedEntity(name="Ana", mention_text="Ana", type="PERSON", subtype="POLITICIAN",
                            evidence_text="Ana pagó.", attributes={"nationality": "ES"},
                            confidence=0.9),
            ExtractedEntity(name="Acme", mention_text="Acme", type="ORGANIZATION", subtype=None,
                            evidence_text="a Acme.", attributes={"jurisdiction": "FR"},
                            confidence=0.8),
        ],
        relations=[
            ExtractedRelation(head=0, tail=1, type="PAYMENT_TO", subtype=None,
                              evidence_text="Ana pagó a Acme.", attributes={"amount": 100},
                              confidence=0.85),
        ],
    )


def _resolve_response():
    return ResolveResponse(
        entities=[
            ResolvedEntity(id="0", canonical_name="Ana", names=["Ana"], type="PERSON",
                           subtype="POLITICIAN", evidence=["Ana pagó."]),
            ResolvedEntity(id="1", canonical_name="Acme", names=["Acme"], type="ORGANIZATION",
                           subtype=None, evidence=["a Acme."]),
        ],
        relations=[
            ResolvedRelation(head_id="0", tail_id="1", type="PAYMENT_TO", subtype=None,
                             evidence=["Ana pagó a Acme."], attributes={"amount": 100},
                             confidence=0.85),
        ],
    )


def _wire_phase_a(monkeypatch, wv):
    monkeypatch.setattr(orch.coref, "resolve_pronouns", lambda t: t)
    monkeypatch.setattr(orch.ner, "run", lambda chunk, schema: _ner_response())
    monkeypatch.setattr(orch.resolver, "resolve", lambda e, r: _resolve_response())
    monkeypatch.setattr(orch.summarizer, "summarize",
                        lambda profile, fields: {"description": f"desc-{profile}"})
    monkeypatch.setattr(orch, "_embed", lambda text: [0.1])
    monkeypatch.setattr(orch.weaviate_io, "Client", lambda use_case: wv)


def test_ingest_document_returns_counts(monkeypatch):
    wv = MagicMock()
    wv.upsert_entity.side_effect = ["e0", "e1"]
    wv.upsert_relation.return_value = "r0"
    _wire_phase_a(monkeypatch, wv)

    res = orch.ingest_document({"doc_id": "d1", "text": "Ana pagó a Acme."}, USE_CASE)

    assert res == {"doc_id": "d1", "entities": 2, "relations": 1}
    wv.ensure_collections.assert_called_once()


def test_ingest_joins_attributes_and_confidence_on_entity_upsert(monkeypatch):
    wv = MagicMock()
    wv.upsert_entity.side_effect = ["e0", "e1"]
    wv.upsert_relation.return_value = "r0"
    _wire_phase_a(monkeypatch, wv)

    orch.ingest_document({"doc_id": "d1", "text": "x"}, USE_CASE)

    ana_props = wv.upsert_entity.call_args_list[0][0][0]
    assert ana_props["canonical_name"] == "Ana"
    assert ana_props["nationality"] == "ES"          # joined from ner attributes
    assert ana_props["confidence"] == 0.9
    assert ana_props["doc_ids"] == ["d1"]
    assert ana_props["aliases"] == ["Ana"]


def test_ingest_remaps_relation_endpoints_to_weaviate_ids(monkeypatch):
    wv = MagicMock()
    wv.upsert_entity.side_effect = ["e0", "e1"]
    wv.upsert_relation.return_value = "r0"
    _wire_phase_a(monkeypatch, wv)

    orch.ingest_document({"doc_id": "d1", "text": "x"}, USE_CASE)

    rel_props = wv.upsert_relation.call_args[0][0]
    assert rel_props["head_id"] == "e0"              # resolved id "0" → weaviate id "e0"
    assert rel_props["tail_id"] == "e1"
    assert rel_props["type"] == "PAYMENT_TO"
    assert json.loads(rel_props["attributes"]) == {"amount": 100}
    assert rel_props["confidence"] == 0.85


def test_ingest_aborts_without_finalising_on_ner_failure(monkeypatch):
    import httpx
    wv = MagicMock()
    _wire_phase_a(monkeypatch, wv)

    def boom(chunk, schema):
        raise httpx.HTTPError("ollama down")
    monkeypatch.setattr(orch.ner, "run", boom)

    res = orch.ingest_document({"doc_id": "d1", "text": "x"}, USE_CASE)
    assert res["status"] == "error"
    wv.upsert_entity.assert_not_called()


# --- Phase B ---

def _wire_phase_b(monkeypatch, wv, final, graph, ent_clusters, rel_clusters):
    monkeypatch.setattr(orch.weaviate_io, "Client", lambda use_case: wv)
    monkeypatch.setattr(orch.weaviate_io, "FinalClient", lambda use_case: final)
    monkeypatch.setattr(orch.neo4j_io, "Graph", lambda use_case: graph)
    monkeypatch.setattr(orch.summarizer, "summarize",
                        lambda profile, fields: {"description": "agg"})
    monkeypatch.setattr(orch, "_embed", lambda text: [0.2])
    results = iter([{"clusters": ent_clusters}, {"clusters": rel_clusters}])
    monkeypatch.setattr(orch.dedup, "dedup_corpus", lambda *a, **k: next(results))


def test_consolidate_merges_clusters_and_writes_final_and_graph(monkeypatch):
    wv = MagicMock()
    wv.get_entity.side_effect = lambda i: {
        "e0": {"canonical_name": "Ana", "type": "PERSON", "subtype": "POLITICIAN",
               "aliases": ["Ana"], "doc_ids": ["d1"], "evidence": ["ev"], "confidence": 0.9,
               "nationality": "ES"},
        "e1": {"canonical_name": "Acme", "type": "ORGANIZATION", "subtype": None,
               "aliases": ["Acme"], "doc_ids": ["d1"], "evidence": ["ev"], "confidence": 0.8,
               "jurisdiction": "FR"},
    }[i]
    wv.get_relation.side_effect = lambda i: {
        "r0": {"head_id": "e0", "tail_id": "e1", "type": "PAYMENT_TO", "subtype": None,
               "evidence": ["ev"], "doc_ids": ["d1"], "confidence": 0.85,
               "attributes": json.dumps({"amount": 100})},
    }[i]

    final = MagicMock()
    final.upsert_entity.side_effect = ["f0", "f1"]
    final.upsert_relation.return_value = "fr0"
    graph = MagicMock()

    _wire_phase_b(monkeypatch, wv, final, graph,
                  ent_clusters=[["e0"], ["e1"]], rel_clusters=[["r0"]])

    res = orch.consolidate(USE_CASE)

    assert res == {"entities": 2, "relations": 1}
    assert final.upsert_entity.call_count == 2
    final.upsert_relation.assert_called_once()
    # relation endpoints remapped through canon (e0→f0, e1→f1)
    edge_args = graph.upsert_edge.call_args[0]
    assert edge_args[1] == "f0" and edge_args[2] == "f1" and edge_args[3] == "PAYMENT_TO"
    assert graph.upsert_node.call_count == 2
