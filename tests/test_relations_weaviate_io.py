import json

import httpx

from pipelines.relations.weaviate_io import Client, FinalClient


def _make(cls, handler):
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://wv:8080")
    return cls("financial_flows", base_url="http://wv:8080", client=http)


def test_client_uses_working_collection_names():
    wv = _make(Client, lambda r: httpx.Response(200, json={}))
    assert wv.entity_collection == "Entity_financial_flows_v1"
    assert wv.relation_collection == "Relation_financial_flows_v1"


def test_final_client_uses_final_collection_names():
    fc = _make(FinalClient, lambda r: httpx.Response(200, json={}))
    assert fc.entity_collection == "Entity_financial_flows_final_v1"
    assert fc.relation_collection == "Relation_financial_flows_final_v1"


def test_ensure_collections_creates_both_when_absent():
    posted = []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(404, json={})
        posted.append(json.loads(request.content)["class"])
        return httpx.Response(200, json={})

    wv = _make(Client, handler)
    wv.ensure_collections()
    assert set(posted) == {"Entity_financial_flows_v1", "Relation_financial_flows_v1"}


def test_upsert_entity_posts_with_vector_and_returns_id():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "e-1"})

    wv = _make(Client, handler)
    eid = wv.upsert_entity({"canonical_name": "X", "type": "PERSON"}, vector=[0.1])
    assert eid == "e-1"
    assert seen["body"]["class"] == "Entity_financial_flows_v1"
    assert seen["body"]["properties"]["canonical_name"] == "X"
    assert seen["body"]["vector"] == [0.1]


def test_upsert_relation_posts_to_relation_collection():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "r-1"})

    wv = _make(Client, handler)
    rid = wv.upsert_relation({"head_id": "e-1", "tail_id": "e-2", "type": "PAYMENT_TO"},
                             vector=[0.2])
    assert rid == "r-1"
    assert seen["body"]["class"] == "Relation_financial_flows_v1"


def test_get_entity_returns_properties():
    def handler(request):
        assert request.url.path == "/v1/objects/Entity_financial_flows_v1/e-1"
        return httpx.Response(200, json={"properties": {"canonical_name": "X", "type": "PERSON"}})

    wv = _make(Client, handler)
    assert wv.get_entity("e-1") == {"canonical_name": "X", "type": "PERSON"}


def test_get_relation_returns_properties():
    def handler(request):
        assert request.url.path == "/v1/objects/Relation_financial_flows_v1/r-1"
        return httpx.Response(200, json={"properties": {"head_id": "e-1", "tail_id": "e-2"}})

    wv = _make(Client, handler)
    assert wv.get_relation("r-1") == {"head_id": "e-1", "tail_id": "e-2"}
