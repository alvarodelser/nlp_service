import os

from transformers import pipeline

_MODEL_NAME = os.environ.get("NLI_MODEL", "Recognai/bert-base-spanish-wwm-cased-xnli")
_pipeline = None


def _ensure_loaded() -> None:
    global _pipeline
    if _pipeline is None:
        _pipeline = pipeline("zero-shot-classification", model=_MODEL_NAME)


def classify(
    text: str,
    labels: list[str],
    multi_label: bool = True,
    hypothesis_template: str = "{}",
) -> dict:
    """Zero-shot NLI classification via the shared XNLI pipeline.

    Returns the raw pipeline dict: {'labels': [...], 'scores': [...], 'sequence': ...}
    sorted descending by score.

    multi_label=True scores each label independently; multi_label=False is the
    mutually-exclusive softmax used for pick-the-best-of-N (scope/city/region selection).
    hypothesis_template: use "{}" when labels are already full hypothesis sentences; use a
    template like "Este artículo trata sobre {}." when labels are short names.
    """
    _ensure_loaded()
    assert _pipeline is not None
    return _pipeline(
        text,
        candidate_labels=labels,
        multi_label=multi_label,
        hypothesis_template=hypothesis_template,
    )


def score(
    text: str,
    hypotheses: list[str],
    threshold: float | None = None,
    blacklist: bool = False,
    hypothesis_template: str = "{}",
) -> list[dict]:
    """Ordered, independently-scored hypotheses with optional short-circuit.

    Returns [{"hypothesis": str, "score": float}, ...] in input order for the hypotheses
    actually run. Scores only — the caller interprets the verdict.

    - threshold None            → run all hypotheses, return every score (caller does OR).
    - threshold set, blacklist=False → stop at the first score BELOW threshold (AND).
    - threshold set, blacklist=True  → stop at the first score AT/ABOVE threshold (AND-exclusion).

    The returned list always includes the hypothesis that triggered the short-circuit.
    """
    if not hypotheses:
        return []

    if threshold is None:
        raw = classify(text, hypotheses, multi_label=True,
                       hypothesis_template=hypothesis_template)
        by_label = dict(zip(raw["labels"], raw["scores"]))      # classify() returns sorted
        return [{"hypothesis": h, "score": float(by_label[h])} for h in hypotheses]

    out: list[dict] = []
    for h in hypotheses:
        raw = classify(text, [h], multi_label=True, hypothesis_template=hypothesis_template)
        s = float(raw["scores"][0])
        out.append({"hypothesis": h, "score": s})
        if not blacklist and s < threshold:        # AND: first failure short-circuits
            break
        if blacklist and s >= threshold:           # AND-exclusion: first violation short-circuits
            break
    return out
