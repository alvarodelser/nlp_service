import os
import re
from dataclasses import dataclass, field

try:
    import torch
    import flair
    from flair.models import SequenceTagger
    from flair.data import Sentence as FlairSentence
    flair.device = torch.device(os.environ.get("NER_DEVICE", "cpu"))
    _FLAIR_AVAILABLE = True
except ImportError:
    _FLAIR_AVAILABLE = False
    SequenceTagger = None
    FlairSentence = None

_KEEP_LABELS = {"LOC"}
_tagger = None

_STREET_RE = re.compile(
    r'(?:^|(?<=\s))'
    r'(?:Calle|Avda?\.?|Avenida|Plaza|Pza\.?|Paseo|Ps\.?|'
    r'Glorieta|Ronda|C/|Camino|Carretera|Ctra\.?)\s+'
    r'([A-ZÁÉÍÓÚÜÑ][^\n,;.]{2,50})',
    re.IGNORECASE,
)


@dataclass
class Span:
    text: str
    label: str
    start_char: int
    end_char: int
    hint: str = ""


def _ensure_loaded() -> None:
    global _tagger
    if _tagger is None:
        if not _FLAIR_AVAILABLE:
            raise RuntimeError(
                "flair is not installed. Add flair>=0.13 to requirements.txt "
                "and rebuild the Docker image."
            )
        _tagger = SequenceTagger.load("flair/ner-spanish-large")


def extract_spans(text: str) -> list[Span]:
    _ensure_loaded()
    assert _tagger is not None

    sentence = FlairSentence(text, use_tokenizer=True)
    _tagger.predict(sentence)

    ner_spans: list[Span] = [
        Span(
            text=entity.text,
            label=entity.tag,
            start_char=entity.start_position,
            end_char=entity.end_position,
        )
        for entity in sentence.get_spans("ner")
        if entity.tag in _KEEP_LABELS
    ]

    covered: set[tuple[int, int]] = {(s.start_char, s.end_char) for s in ner_spans}

    street_spans: list[Span] = []
    for m in _STREET_RE.finditer(text):
        start, end = m.start(), m.end()
        if any(s <= start < e or s < end <= e for s, e in covered):
            continue
        street_spans.append(Span(
            text=m.group(0).strip(),
            label="LOC",
            start_char=start,
            end_char=end,
            hint="street",
        ))
        covered.add((start, end))

    return ner_spans + street_spans
