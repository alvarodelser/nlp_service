import json

import httpx
import pytest

from pipelines.news.weaviate_io import Client


def _client(handler):
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport, base_url="http://wv:8080")
    return Client("Article_news_v1", base_url="http://wv:8080", client=http)


def test_upsert_posts_object_with_vector_and_returns_id():
    seen = {}

    def handler(request):
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "new-uuid"})

    wv = _client(handler)
    obj_id = wv.upsert({"headline": "h", "summary": "s"}, vector=[0.1, 0.2])

    assert obj_id == "new-uuid"
    assert seen["method"] == "POST"
    assert seen["url"].endswith("/v1/objects")
    assert seen["body"]["class"] == "Article_news_v1"
    assert seen["body"]["properties"] == {"headline": "h", "summary": "s"}
    assert seen["body"]["vector"] == [0.1, 0.2]


def test_get_returns_only_requested_props():
    def handler(request):
        assert request.method == "GET"
        assert request.url.path == "/v1/objects/Article_news_v1/obj-1"
        return httpx.Response(200, json={
            "properties": {"sources": ["a"], "urls": ["u"], "date": "2026-01-01",
                           "summary": "ignored"},
        })

    wv = _client(handler)
    out = wv.get("obj-1", return_props=("sources", "urls", "date"))
    assert out == {"sources": ["a"], "urls": ["u"], "date": "2026-01-01"}


def test_update_patches_properties():
    seen = {}

    def handler(request):
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={})

    wv = _client(handler)
    wv.update("obj-1", {"sources": ["a", "b"]})

    assert seen["method"] == "PATCH"
    assert seen["path"] == "/v1/objects/Article_news_v1/obj-1"
    assert seen["body"]["properties"] == {"sources": ["a", "b"]}


def test_ensure_collection_creates_when_absent():
    calls = []

    def handler(request):
        calls.append((request.method, request.url.path))
        if request.method == "GET":
            return httpx.Response(404, json={})
        return httpx.Response(200, json={})   # POST /v1/schema

    wv = _client(handler)
    wv.ensure_collection()

    assert ("GET", "/v1/schema/Article_news_v1") in calls
    assert ("POST", "/v1/schema") in calls


def test_ensure_collection_noop_when_present():
    calls = []

    def handler(request):
        calls.append(request.method)
        return httpx.Response(200, json={"class": "Article_news_v1"})

    wv = _client(handler)
    wv.ensure_collection()
    assert calls == ["GET"]   # no POST


def test_iter_recent_yields_objects_with_uuid_and_properties():
    def handler(request):
        assert request.url.path == "/v1/graphql"
        body = {
            "data": {"Get": {"Article_news_v1": [
                {"minhash": [1, 2, 3], "_additional": {"id": "u1"}},
                {"minhash": [4, 5, 6], "_additional": {"id": "u2"}},
            ]}}
        }
        return httpx.Response(200, json=body)

    wv = _client(handler)
    objs = list(wv.iter_recent(return_props=("minhash",), limit=10))
    assert [o.uuid for o in objs] == ["u1", "u2"]
    assert objs[0].properties["minhash"] == [1, 2, 3]
