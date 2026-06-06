import json

from pipelines.relations.merge import _merge_entity_cluster, _merge_relation_cluster


def test_merge_entity_picks_longest_name_and_unions():
    members = [
        {"canonical_name": "Banco", "type": "ORGANIZATION", "subtype": None,
         "aliases": ["Banco"], "doc_ids": ["d1"], "evidence": ["e1"], "confidence": 0.8,
         "jurisdiction": "FR"},
        {"canonical_name": "Banco Santander", "type": "ORGANIZATION", "subtype": "SHELL_COMPANY",
         "aliases": ["Banco Santander", "BS"], "doc_ids": ["d2"], "evidence": ["e2"],
         "confidence": 0.9, "jurisdiction": "ES"},
    ]
    merged = _merge_entity_cluster(members, ["jurisdiction"])
    assert merged["canonical_name"] == "Banco Santander"
    assert merged["type"] == "ORGANIZATION"
    assert merged["subtype"] == "SHELL_COMPANY"
    assert merged["aliases"] == ["BS", "Banco", "Banco Santander"]      # sorted union
    assert merged["doc_ids"] == ["d1", "d2"]
    assert set(merged["evidence"]) == {"e1", "e2"}
    assert merged["jurisdiction"] == "ES"                              # 0.9 confidence wins
    assert merged["confidence"] == 0.9


def test_merge_entity_attribute_falls_through_to_lower_confidence():
    members = [
        {"canonical_name": "X", "type": "PERSON", "subtype": None, "aliases": [], "doc_ids": [],
         "evidence": [], "confidence": 0.9, "nationality": None},
        {"canonical_name": "X long", "type": "PERSON", "subtype": None, "aliases": [],
         "doc_ids": [], "evidence": [], "confidence": 0.4, "nationality": "ES"},
    ]
    merged = _merge_entity_cluster(members, ["nationality"])
    assert merged["nationality"] == "ES"


def test_merge_relation_remaps_endpoints_and_unions_attributes():
    members = [
        {"head_id": "w1", "tail_id": "w2", "type": "PAYMENT_TO", "subtype": None,
         "evidence": ["e1"], "doc_ids": ["d1"], "confidence": 0.9,
         "attributes": json.dumps({"amount": 100})},
        {"head_id": "w1", "tail_id": "w2", "type": "PAYMENT_TO", "subtype": None,
         "evidence": ["e2"], "doc_ids": ["d2"], "confidence": 0.8,
         "attributes": json.dumps({"amount": 150, "currency": "EUR"})},
    ]
    canon = {"w1": "f1", "w2": "f2"}
    merged = _merge_relation_cluster(members, ["amount", "currency"], canon)
    assert merged["head_id"] == "f1"
    assert merged["tail_id"] == "f2"
    assert merged["type"] == "PAYMENT_TO"
    assert merged["attributes"] == {"amount": 100, "currency": "EUR"}   # amount: 0.9 wins
    assert set(merged["evidence"]) == {"e1", "e2"}
    assert merged["doc_ids"] == ["d1", "d2"]
    assert merged["confidence"] == 0.9
