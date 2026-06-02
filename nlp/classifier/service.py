# nlp_service/nlp/classifier/service.py
from __future__ import annotations

from . import model, taxonomy


def load() -> None:
    taxonomy.load()


def run(
    summary: str,
    geo_cities: list[dict] | None = None,
    search_tags: list[str] | None = None,
    source_profile: dict | None = None,
    geo_scope: str | None = None,
) -> dict:
    tax = taxonomy.load()
    geo_cities = geo_cities or []
    search_tags = search_tags or []

    # Relevance gate: NLI hypothesis check on the summary
    if tax.relevance_hypothesis:
        rel = model.classify(summary, labels=[tax.relevance_hypothesis], multi_label=True)
        rel_score = rel["scores"][0] if rel["scores"] else 0.0
        if rel_score < tax.relevance_threshold:
            return {"topics": [], "scores": {}, "geo_scope": geo_scope or "national", "out_of_scope": True}

    # Topic NLI: premise is search_tags + summary (search tags act as editorial prior)
    tag_prefix = ""
    if search_tags:
        tag_prefix = "Artículo buscado por: " + ", ".join(f"'{t}'" for t in search_tags) + ". "
    topic_premise = tag_prefix + summary

    all_labels = tax.labels + tax.blacklist_labels
    raw = model.classify(
        topic_premise,
        labels=all_labels,
        multi_label=True,
        hypothesis_template="Este texto trata sobre {}.",
    )
    scored = dict(zip(raw["labels"], raw["scores"]))

    if tax.blacklist_labels:
        top_blacklist_score = max(scored.get(lbl, 0.0) for lbl in tax.blacklist_labels)
        if top_blacklist_score >= tax.blacklist_threshold:
            return {"topics": [], "scores": scored, "geo_scope": geo_scope or "national", "out_of_scope": True}

    filtered = sorted(
        [(lbl, scored[lbl]) for lbl in tax.labels if scored.get(lbl, 0) >= tax.score_threshold],
        key=lambda x: x[1], reverse=True,
    )[:tax.top_k]

    # Scope NLI: separate 3-way exclusive pass with assembled geographic evidence
    resolved_scope = geo_scope or _scope_pass(summary, geo_cities, source_profile, tax)

    return {
        "topics": [lbl for lbl, _ in filtered],
        "scores": {lbl: scored[lbl] for lbl in tax.labels},
        "geo_scope": resolved_scope,
        "out_of_scope": False,
    }


def _scope_pass(
    summary: str,
    geo_cities: list[dict],
    source_profile: dict | None,
    tax: taxonomy.Taxonomy,
) -> str:
    if not tax.scope_hypotheses:
        return _scope_fallback(geo_cities)

    city_context = ""
    if geo_cities:
        names = ", ".join(c["city_name"] for c in geo_cities[:5])
        city_context = f" Se mencionan las ciudades: {names}."

    source_context = ""
    if source_profile:
        if source_profile.get("city"):
            source_context += f" La fuente cubre habitualmente: {source_profile['city']}."
        elif source_profile.get("region"):
            source_context += f" La fuente cubre habitualmente: {source_profile['region']}."

    premise = summary + city_context + source_context

    scope_scores: dict[str, float] = {}
    for label, hyp in tax.scope_hypotheses.items():
        result = model.classify(premise, labels=[hyp], multi_label=False)
        scope_scores[label] = result["scores"][0]

    best_scope = max(scope_scores, key=lambda k: scope_scores[k])
    if scope_scores[best_scope] >= tax.scope_threshold:
        return best_scope

    return _scope_fallback(geo_cities)


def _scope_fallback(geo_cities: list[dict]) -> str:
    if len(geo_cities) > 1:
        return "regional"
    if len(geo_cities) == 1:
        return "city"
    return "national"
