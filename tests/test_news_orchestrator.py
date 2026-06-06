import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from pipelines.news import orchestrator as orch
from pipelines.news.weaviate_io import RecentObject


def _cfg():
    return SimpleNamespace(collection="Article_news_v1")


def _article():
    return {"title": "T", "text": "cuerpo del articulo", "source": "elpais",
            "url": "http://a", "date": "2026-01-01"}


def _wire(monkeypatch, **over):
    """Patch every external seam of run_article to a happy 'indexed' default; override per test."""
    wv = over.get("wv") or MagicMock()
    if "upsert" not in [c for c in dir(wv)]:
        pass
    wv.upsert.return_value = over.get("upsert_id", "obj-123")
    merge_spy = MagicMock()

    monkeypatch.setattr(orch.cfg_mod, "load", lambda: _cfg())
    monkeypatch.setattr(orch.weaviate_io, "Client", lambda collection: wv)
    monkeypatch.setattr(orch.minhash, "compute", lambda text: [1, 2, 3])
    monkeypatch.setattr(orch, "_minhash_match", over.get("minhash_match", lambda wv, sig: None))
    monkeypatch.setattr(orch.summarizer, "summarize",
                        over.get("summarize", lambda profile, fields: {"headline": "H", "summary": "S"}))
    monkeypatch.setattr(orch, "_in_scope", over.get("in_scope", lambda text, cfg: True))
    monkeypatch.setattr(orch, "_embed", over.get("embed", lambda text: [0.1, 0.2]))
    monkeypatch.setattr(orch.dedup_service, "dedup_item",
                        over.get("dedup_item",
                                 lambda *a, **k: {"decision": "no_match", "target_id": None}))
    monkeypatch.setattr(orch.geotagger, "run",
                        over.get("geotag", lambda text, headline="": {"places": []}))
    monkeypatch.setattr(orch, "_dedupe_places", lambda places: places)
    monkeypatch.setattr(orch, "_assign_topics", over.get("topics", lambda text, cfg: ["movilidad"]))
    monkeypatch.setattr(orch, "_assign_scope",
                        over.get("scope", lambda text, places, cfg: ("city", None, [10])))
    monkeypatch.setattr(orch.merge_mod, "merge_article", merge_spy)
    return SimpleNamespace(wv=wv, merge=merge_spy)


def test_indexed_happy_path(monkeypatch):
    w = _wire(monkeypatch)
    res = orch.run_article(_article())
    assert res["status"] == "indexed"
    assert res["id"] == "obj-123"
    assert res["topics"] == ["movilidad"]
    assert res["scope"] == "city"
    w.wv.upsert.assert_called_once()
    w.merge.assert_not_called()


def test_indexed_upsert_payload_shape(monkeypatch):
    w = _wire(monkeypatch)
    orch.run_article(_article())
    props, kwargs = w.wv.upsert.call_args[0], w.wv.upsert.call_args[1]
    payload = props[0]
    assert payload["headline"] == "H"
    assert payload["summary"] == "S"
    assert payload["sources"] == ["elpais"]
    assert payload["urls"] == ["http://a"]
    assert payload["minhash"] == [1, 2, 3]
    assert payload["topics"] == ["movilidad"]
    assert payload["scope"] == "city"
    assert payload["city_ids"] == [10]
    assert kwargs["vector"] == [0.1, 0.2]


def test_minhash_duplicate_merges_and_stops(monkeypatch):
    w = _wire(monkeypatch, minhash_match=lambda wv, sig: "dup-1")
    res = orch.run_article(_article())
    assert res["status"] == "duplicate"
    assert res["stage"] == "minhash"
    w.merge.assert_called_once()
    assert w.merge.call_args[0][1] == "dup-1"
    w.wv.upsert.assert_not_called()


def test_out_of_scope_stops_before_embedding(monkeypatch):
    embed_spy = MagicMock(return_value=[0.0])
    w = _wire(monkeypatch, in_scope=lambda text, cfg: False, embed=embed_spy)
    res = orch.run_article(_article())
    assert res["status"] == "out_of_scope"
    embed_spy.assert_not_called()
    w.wv.upsert.assert_not_called()


def test_embedding_duplicate_merges_and_stops(monkeypatch):
    w = _wire(monkeypatch,
              dedup_item=lambda *a, **k: {"decision": "match", "target_id": "dup-9"})
    res = orch.run_article(_article())
    assert res["status"] == "duplicate"
    assert res["stage"] == "embedding"
    w.merge.assert_called_once()
    assert w.merge.call_args[0][1] == "dup-9"
    w.wv.upsert.assert_not_called()


def test_summarizer_failure_aborts_without_write(monkeypatch):
    def boom(profile, fields):
        raise httpx.HTTPError("ollama down")
    w = _wire(monkeypatch, summarize=boom)
    res = orch.run_article(_article())
    assert res["status"] == "error"
    assert res["stage"] == "summarize"
    w.wv.upsert.assert_not_called()
    w.merge.assert_not_called()


# --- _minhash_match (real) ---

def test_minhash_match_returns_uuid_on_high_jaccard(monkeypatch):
    wv = MagicMock()
    wv.iter_recent.return_value = [RecentObject(uuid="u1", properties={"minhash": [1, 2, 3]})]
    monkeypatch.setenv("NEWS_MINHASH_THRESHOLD", "0.9")
    assert orch._minhash_match(wv, [1, 2, 3]) == "u1"


def test_minhash_match_returns_none_when_below_threshold():
    wv = MagicMock()
    wv.iter_recent.return_value = [RecentObject(uuid="u1", properties={"minhash": [9, 9, 9]})]
    assert orch._minhash_match(wv, [1, 2, 3]) is None


# --- _embed (real, mocked HTTP) ---

def test_embed_posts_text_and_returns_embedding(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return httpx.Response(200, json={"embedding": [0.5, 0.6]},
                              request=httpx.Request("POST", url))
    monkeypatch.setattr(orch.httpx, "post", fake_post)

    vec = orch._embed("hola mundo")
    assert vec == [0.5, 0.6]
    assert captured["json"] == {"text": "hola mundo"}
