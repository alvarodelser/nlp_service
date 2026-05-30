import re
from dataclasses import dataclass
from transformers import pipeline

NER_MODEL = "PlanTL-GOB-ES/roberta-base-bne-ner-capiter"
STREET_PREFIX = re.compile(
    r"^(Calle|Avda?\.?|Avenida|Plaza|Paseo|Ronda|Travesía|Carretera|C/)\s+",
    re.IGNORECASE,
)

_ner_pipeline = None


@dataclass
class Span:
    text: str
    label: str       # LOC | GPE | FAC
    hint: str | None  # "street" if street prefix detected
    char_start: int
    sentence: str    # sentence the span appears in (for co-occurrence)


def startup() -> None:
    global _ner_pipeline
    _ner_pipeline = pipeline(
        "token-classification",
        model=NER_MODEL,
        aggregation_strategy="simple",
    )


def extract_spans(text: str, headline: str) -> list[Span]:
    full = headline + ". " + text
    raw = _ner_pipeline(full)
    spans: list[Span] = []
    sentences = _split_sentences(full)

    for entity in raw:
        label = entity["entity_group"]
        if label not in ("LOC", "GPE", "FAC"):
            continue
        span_text = entity["word"].strip()
        char_start = entity["start"]
        hint = "street" if STREET_PREFIX.match(span_text) else None
        sentence = _find_sentence(sentences, char_start)
        spans.append(Span(span_text, label, hint, char_start, sentence))

    return spans


def _split_sentences(text: str) -> list[tuple[int, str]]:
    import re
    result = []
    for m in re.finditer(r"[^.!?]+[.!?]?", text):
        result.append((m.start(), m.group()))
    return result


def _find_sentence(sentences: list[tuple[int, str]], char_start: int) -> str:
    for start, sent in reversed(sentences):
        if start <= char_start:
            return sent
    return ""
