import textwrap

import pytest

from pipelines.relations import ontology
from nlp.ner.schema_types import ExtractionSchema


# --- shipped financial_flows ontology ---

def test_load_shipped_financial_flows():
    data = ontology.load("financial_flows")
    assert data["use_case"] == "financial_flows"
    assert data["version"] == "1"
    assert [e["name"] for e in data["entity_types"]] == ["PERSON", "ORGANIZATION"]


def test_extraction_schema_projection():
    schema = ontology.extraction_schema("financial_flows")
    assert isinstance(schema, ExtractionSchema)
    person = next(e for e in schema.entity_types if e.name == "PERSON")
    assert {s.name for s in person.subtypes} == {"POLITICIAN", "EXECUTIVE"}
    assert {a.name for a in person.attributes} == {"nationality", "role_title"}
    rel = schema.relation_types[0]
    assert rel.name == "PAYMENT_TO"
    assert rel.head_types == ["PERSON", "ORGANIZATION"]
    assert {a.name for a in rel.attributes} == {"amount", "currency"}


def test_entity_spec_name_and_attribute_columns():
    spec = ontology.weaviate_entity_spec("financial_flows")
    assert spec["class"] == "Entity_financial_flows_v1"
    names = {p["name"] for p in spec["properties"]}
    assert {"canonical_name", "type", "subtype", "description", "aliases",
            "doc_ids", "evidence", "confidence"} <= names
    assert {"nationality", "role_title", "jurisdiction"} <= names   # union across entity types


def test_relation_spec_name_and_columns():
    spec = ontology.weaviate_relation_spec("financial_flows")
    assert spec["class"] == "Relation_financial_flows_v1"
    props = {p["name"]: p["dataType"] for p in spec["properties"]}
    assert "head_id" in props and "tail_id" in props
    assert props["attributes"] == ["text"]            # relation attributes kept as one JSON blob
    assert {"head_id", "tail_id", "type", "subtype", "description",
            "evidence", "doc_ids", "confidence"} <= set(props)


def test_final_spec_names_carry_final():
    assert ontology.weaviate_entity_spec("financial_flows", final=True)["class"] \
        == "Entity_financial_flows_final_v1"
    assert ontology.weaviate_relation_spec("financial_flows", final=True)["class"] \
        == "Relation_financial_flows_final_v1"


# --- validation ---

def _write(tmp_path, monkeypatch, body, use_case="uc_test"):
    monkeypatch.setenv("ONTOLOGY_DIR", str(tmp_path))
    (tmp_path / f"{use_case}.yaml").write_text(textwrap.dedent(body), encoding="utf-8")
    return use_case


_BASE = """
    version: "1"
    use_case: uc_test
    entity_types:
      - name: PERSON
        attributes:
          nationality: {{type: string}}
    relation_types:
      - name: PAYMENT_TO
        head_types: [PERSON]
        tail_types: [{tail}]
        attributes:
          amount: {{type: {dtype}}}
"""


def test_valid_minimal_ontology_loads(tmp_path, monkeypatch):
    uc = _write(tmp_path, monkeypatch, _BASE.format(tail="PERSON", dtype="number"))
    assert ontology.load(uc)["use_case"] == "uc_test"


def test_rejects_lowercase_entity_name(tmp_path, monkeypatch):
    body = _BASE.format(tail="PERSON", dtype="number").replace("name: PERSON", "name: person")
    uc = _write(tmp_path, monkeypatch, body)
    with pytest.raises(ValueError):
        ontology.load(uc)


def test_rejects_undeclared_relation_endpoint_type(tmp_path, monkeypatch):
    uc = _write(tmp_path, monkeypatch, _BASE.format(tail="GHOST", dtype="number"))
    with pytest.raises(ValueError):
        ontology.load(uc)


def test_rejects_invalid_attribute_datatype(tmp_path, monkeypatch):
    uc = _write(tmp_path, monkeypatch, _BASE.format(tail="PERSON", dtype="float64"))
    with pytest.raises(ValueError):
        ontology.load(uc)
