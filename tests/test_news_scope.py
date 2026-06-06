from types import SimpleNamespace

import pytest

from nlp.geotagger.service import GeoEntity
from pipelines.news import orchestrator as orch


# --- _in_scope ---

def _scope_cfg():
    return SimpleNamespace(
        in_scope_hyps=["a", "b"], in_scope_threshold=0.40,
        blacklist_hyps=["x"], blacklist_threshold=0.70,
    )


def _patch_score(monkeypatch, whitelist_scores, blacklist_scores):
    def fake_score(text, hyps, threshold=None, blacklist=False, hypothesis_template="{}"):
        scores = blacklist_scores if blacklist else whitelist_scores
        return [{"hypothesis": h, "score": s} for h, s in zip(hyps, scores)]
    monkeypatch.setattr(orch.nli, "score", fake_score)


def test_in_scope_rejects_when_no_whitelist_hypothesis_clears(monkeypatch):
    _patch_score(monkeypatch, whitelist_scores=[0.1, 0.2], blacklist_scores=[0.0])
    assert orch._in_scope("t", _scope_cfg()) is False


def test_in_scope_accepts_when_whitelist_clears_and_blacklist_clean(monkeypatch):
    _patch_score(monkeypatch, whitelist_scores=[0.1, 0.9], blacklist_scores=[0.1])
    assert orch._in_scope("t", _scope_cfg()) is True


def test_in_scope_rejects_when_blacklist_hit(monkeypatch):
    _patch_score(monkeypatch, whitelist_scores=[0.9], blacklist_scores=[0.95])
    assert orch._in_scope("t", _scope_cfg()) is False


# --- _assign_topics ---

def test_assign_topics_keeps_above_threshold_and_truncates_top_k(monkeypatch):
    cfg = SimpleNamespace(topic_labels=["m", "v", "p", "s"], topic_threshold=0.50,
                          topic_top_k=2, topic_template="Sobre {}.")

    def fake_score(text, hyps, threshold=None, blacklist=False, hypothesis_template="{}"):
        scoremap = {"m": 0.9, "v": 0.8, "p": 0.55, "s": 0.2}
        return [{"hypothesis": h, "score": scoremap[h]} for h in hyps]
    monkeypatch.setattr(orch.nli, "score", fake_score)

    assert orch._assign_topics("t", cfg) == ["m", "v"]   # s dropped (<0.5), p cut by top_k


def test_assign_topics_empty_when_none_clear(monkeypatch):
    cfg = SimpleNamespace(topic_labels=["m"], topic_threshold=0.50, topic_top_k=3,
                          topic_template="{}")
    monkeypatch.setattr(orch.nli, "score",
                        lambda *a, **k: [{"hypothesis": "m", "score": 0.1}])
    assert orch._assign_topics("t", cfg) == []


# --- _assign_scope ---

def _scope_assign_cfg():
    return SimpleNamespace(
        scope_levels={"city": "HC", "regional": "HR", "national": "HN"},
        scope_threshold=0.35, place_template="Sobre {}.",
    )


def test_assign_scope_single_city_strong(monkeypatch):
    places = [GeoEntity(text="Madrid", type="city", city_id=10, city_name="Madrid",
                        lat=40.4, lon=-3.7)]
    # pass 1 picks the city hypothesis strongly
    monkeypatch.setattr(orch.nli, "classify",
                        lambda *a, **k: {"labels": ["HC", "HR", "HN"], "scores": [0.9, 0.05, 0.05]})
    scope, region, city_ids = orch._assign_scope("t", places, _scope_assign_cfg())
    assert scope == "city"
    assert region is None
    assert city_ids == [10]


def test_assign_scope_multiple_cities_runs_pass2(monkeypatch):
    places = [
        GeoEntity(text="Madrid", type="city", city_id=10, city_name="Madrid", lat=40.4, lon=-3.7),
        GeoEntity(text="Sevilla", type="city", city_id=20, city_name="Sevilla", lat=37.4, lon=-6.0),
    ]
    calls = []

    def fake_classify(text, labels, multi_label=True, hypothesis_template="{}"):
        calls.append(labels)
        if len(calls) == 1:                       # pass 1 — level pick
            return {"labels": ["HC", "HR", "HN"], "scores": [0.8, 0.1, 0.1]}
        return {"labels": ["Sevilla", "Madrid"], "scores": [0.7, 0.3]}   # pass 2 — city pick
    monkeypatch.setattr(orch.nli, "classify", fake_classify)

    scope, region, city_ids = orch._assign_scope("t", places, _scope_assign_cfg())
    assert scope == "city"
    assert city_ids == [20]                       # Sevilla won pass 2


def test_assign_scope_weak_score_falls_back_to_geometry(monkeypatch):
    places = [GeoEntity(text="Madrid", type="city", city_id=10, city_name="Madrid",
                        lat=40.4, lon=-3.7)]
    # national wins but below threshold → geometry fallback to city (a city is present)
    monkeypatch.setattr(orch.nli, "classify",
                        lambda *a, **k: {"labels": ["HN", "HR", "HC"], "scores": [0.2, 0.1, 0.1]})
    scope, _, city_ids = orch._assign_scope("t", places, _scope_assign_cfg())
    assert scope == "city"
    assert city_ids == [10]


def test_assign_scope_regional_runs_pass2_for_region(monkeypatch):
    places = [GeoEntity(text="Cataluña", type="region", name="Cataluña", geonames_id=1)]
    calls = []

    def fake_classify(text, labels, multi_label=True, hypothesis_template="{}"):
        calls.append(labels)
        if len(calls) == 1:
            return {"labels": ["HR", "HC", "HN"], "scores": [0.8, 0.1, 0.1]}
        return {"labels": ["Cataluña"], "scores": [0.9]}
    monkeypatch.setattr(orch.nli, "classify", fake_classify)

    scope, region, city_ids = orch._assign_scope("t", places, _scope_assign_cfg())
    assert scope == "regional"
    assert region == "Cataluña"
    assert city_ids == []


def test_assign_scope_national_no_places(monkeypatch):
    monkeypatch.setattr(orch.nli, "classify",
                        lambda *a, **k: {"labels": ["HN", "HR", "HC"], "scores": [0.9, 0.05, 0.05]})
    scope, region, city_ids = orch._assign_scope("t", [], _scope_assign_cfg())
    assert scope == "national"
    assert region is None
    assert city_ids == []
