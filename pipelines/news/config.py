"""News-pipeline configuration. All domain taxonomy/threshold knowledge lives in
`config/topics.yaml`; this loads it into a frozen NewsConfig. The agnostic nlp modules know
none of this."""
import os
from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class NewsConfig:
    in_scope_hyps: list[str]
    in_scope_threshold: float
    blacklist_hyps: list[str]
    blacklist_threshold: float
    topic_labels: list[str]
    topic_threshold: float
    topic_top_k: int
    topic_template: str
    scope_levels: dict[str, str]
    scope_threshold: float
    place_template: str
    collection: str


def load(path: str | None = None) -> NewsConfig:
    src = path or os.environ.get("TOPICS_YAML_PATH", "config/topics.yaml")
    with open(src, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return NewsConfig(
        in_scope_hyps=list(data["in_scope"]["hypotheses"]),
        in_scope_threshold=float(data["in_scope"]["threshold"]),
        blacklist_hyps=list(data["blacklist"]["hypotheses"]),
        blacklist_threshold=float(data["blacklist"]["threshold"]),
        topic_labels=list(data["topics"]["labels"]),
        topic_threshold=float(data["topics"]["threshold"]),
        topic_top_k=int(data["topics"]["top_k"]),
        topic_template=data["topics"]["hypothesis_template"],
        scope_levels=dict(data["scope"]["level_hypotheses"]),
        scope_threshold=float(data["scope"]["threshold"]),
        place_template=data["scope"]["place_template"],
        collection=os.environ.get("NEWS_COLLECTION", "Article_news_v1"),
    )
