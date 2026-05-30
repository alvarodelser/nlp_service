"""
Two separate NLI passes:
  1. Topic pass  — multi-label, premise = search_tags + summary
  2. Scope pass  — 3-way exclusive, premise = summary + city evidence + source profile
"""

from transformers import pipeline
from nlp.classifier import taxonomy
from api.models import ClassifyResponse, ResolvedCity, SourceProfile

NLI_MODEL = "Recognai/bert-base-spanish-wwm-cased-xnli"

_nli = None


def startup() -> None:
    global _nli
    taxonomy.startup()
    _nli = pipeline("zero-shot-classification", model=NLI_MODEL)


def classify(
    summary: str,
    geo_cities: list[ResolvedCity],
    search_tags: list[str],
    source_profile: SourceProfile | None,
) -> ClassifyResponse:
    topics, scores = _topic_pass(summary, search_tags)
    geo_scope = _scope_pass(summary, geo_cities, source_profile)
    return ClassifyResponse(topics=topics, scores=scores, geo_scope=geo_scope)


def _topic_pass(summary: str, search_tags: list[str]) -> tuple[list[str], dict[str, float]]:
    tag_prefix = ""
    if search_tags:
        tag_prefix = "Artículo buscado por: " + ", ".join(f"'{t}'" for t in search_tags) + ". "
    premise = tag_prefix + summary

    result = _nli(
        premise,
        candidate_labels=taxonomy.labels(),
        multi_label=True,
        hypothesis_template="Este artículo trata sobre {}.",
    )

    threshold = taxonomy.nli_threshold()
    scores = dict(zip(result["labels"], result["scores"]))
    topics = [label for label, score in scores.items() if score >= threshold]
    return topics, scores


def _scope_pass(
    summary: str,
    geo_cities: list[ResolvedCity],
    source_profile: SourceProfile | None,
) -> str:
    city_context = ""
    if geo_cities:
        names = ", ".join(c.city_name for c in geo_cities[:5])
        city_context = f" Se mencionan las ciudades: {names}."
    source_context = ""
    if source_profile:
        if source_profile.city:
            source_context += f" La fuente cubre habitualmente: {source_profile.city}."
        elif source_profile.region:
            source_context += f" La fuente cubre habitualmente: {source_profile.region}."

    premise = summary + city_context + source_context

    hypotheses = taxonomy.scope_hypotheses()
    candidate_labels = list(hypotheses.keys())    # national, regional, city
    hypothesis_texts = list(hypotheses.values())

    # Run NLI once per hypothesis and pick the highest-scoring one above threshold
    scope_scores: dict[str, float] = {}
    for label, hyp in zip(candidate_labels, hypothesis_texts):
        result = _nli(premise, candidate_labels=[hyp], multi_label=False)
        scope_scores[label] = result["scores"][0]

    threshold = taxonomy.scope_threshold()
    best_scope = max(scope_scores, key=scope_scores.get)
    if scope_scores[best_scope] >= threshold:
        return best_scope

    # Fallback: infer from city evidence alone
    if len(geo_cities) > 1:
        return "regional"
    if len(geo_cities) == 1:
        return "city"
    return "national"
