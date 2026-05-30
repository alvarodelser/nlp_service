import os
import yaml

TOPICS_PATH = os.getenv("TOPICS_YAML_PATH", "config/topics.yaml")

_config: dict = {}


def startup() -> None:
    global _config
    with open(TOPICS_PATH, encoding="utf-8") as f:
        _config = yaml.safe_load(f)


def labels() -> list[str]:
    return _config.get("labels", [])


def nli_threshold() -> float:
    return float(_config.get("nli_threshold", 0.30))


def scope_threshold() -> float:
    return float(_config.get("scope_threshold", 0.35))


def scope_hypotheses() -> dict[str, str]:
    return _config.get("scope_hypotheses", {})
