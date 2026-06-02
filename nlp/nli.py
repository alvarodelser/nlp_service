from transformers import pipeline

_MODEL_NAME = "Recognai/bert-base-spanish-wwm-cased-xnli"
_pipeline = None


def _ensure_loaded() -> None:
    global _pipeline
    if _pipeline is None:
        _pipeline = pipeline("zero-shot-classification", model=_MODEL_NAME)


def classify(
    text: str,
    labels: list[str],
    multi_label: bool,
    hypothesis_template: str = "{}",
) -> dict:
    """Zero-shot NLI classification via the shared XNLI pipeline.

    Returns the raw pipeline dict: {'labels': [...], 'scores': [...], 'sequence': ...}
    sorted descending by score.

    hypothesis_template: use "{}" when labels are already full hypothesis sentences
    (classifier topic/relevance calls). Use a template like
    "Este artículo trata sobre {}." when labels are short names (city names, etc.).
    """
    _ensure_loaded()
    assert _pipeline is not None
    return _pipeline(
        text,
        candidate_labels=labels,
        multi_label=multi_label,
        hypothesis_template=hypothesis_template,
    )
