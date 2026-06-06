import textwrap

from pipelines.news import config as cfg_mod


_YAML = textwrap.dedent("""
    in_scope:
      threshold: 0.40
      hypotheses:
        - "Trata sobre urbanismo."
        - "Trata sobre política municipal."
    blacklist:
      threshold: 0.70
      hypotheses:
        - "Trata sobre deportes."
    topics:
      threshold: 0.50
      top_k: 3
      labels:
        - "movilidad"
        - "vivienda"
      hypothesis_template: "Este texto trata sobre {}."
    scope:
      threshold: 0.35
      level_hypotheses:
        city: "Actuaciones de un ayuntamiento."
        regional: "Políticas de una comunidad autónoma."
        national: "Asunto de alcance estatal."
      place_template: "Trata sobre {}."
""")


def _write(tmp_path):
    p = tmp_path / "topics.yaml"
    p.write_text(_YAML, encoding="utf-8")
    return str(p)


def test_load_parses_all_sections(tmp_path):
    cfg = cfg_mod.load(_write(tmp_path))
    assert cfg.in_scope_hyps == ["Trata sobre urbanismo.", "Trata sobre política municipal."]
    assert cfg.in_scope_threshold == 0.40
    assert cfg.blacklist_hyps == ["Trata sobre deportes."]
    assert cfg.blacklist_threshold == 0.70
    assert cfg.topic_labels == ["movilidad", "vivienda"]
    assert cfg.topic_threshold == 0.50
    assert cfg.topic_top_k == 3
    assert cfg.topic_template == "Este texto trata sobre {}."
    assert cfg.scope_levels == {
        "city": "Actuaciones de un ayuntamiento.",
        "regional": "Políticas de una comunidad autónoma.",
        "national": "Asunto de alcance estatal.",
    }
    assert cfg.scope_threshold == 0.35
    assert cfg.place_template == "Trata sobre {}."


def test_collection_defaults_when_env_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("NEWS_COLLECTION", raising=False)
    cfg = cfg_mod.load(_write(tmp_path))
    assert cfg.collection == "Article_news_v1"


def test_collection_honors_env(tmp_path, monkeypatch):
    monkeypatch.setenv("NEWS_COLLECTION", "Article_test")
    cfg = cfg_mod.load(_write(tmp_path))
    assert cfg.collection == "Article_test"


def test_shipped_topics_yaml_loads_with_expected_scope_levels():
    # the real config/topics.yaml must satisfy the schema the orchestrator expects
    cfg = cfg_mod.load("config/topics.yaml")
    assert set(cfg.scope_levels) == {"city", "regional", "national"}
    assert cfg.in_scope_hyps and cfg.blacklist_hyps and cfg.topic_labels
    assert "{}" in cfg.topic_template and "{}" in cfg.place_template
