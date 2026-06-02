# Geotagger Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken NER model and rule-based city disambiguation with Flair NER + 2-level NLI geo-classification; wire up TextRank extraction; share the NLI model across geotagger and classifier.

**Architecture:** Flair's document-level NER produces clean LOC spans → gazetteer splits them into city/region/street candidates → two sequential NLI calls (scope first, then city/region) replace all hand-written scoring. TextRank replaces TF-IDF in the extractor. A new `nlp/nli.py` singleton is shared by both classifier and geotagger, eliminating cross-module coupling.

**Tech Stack:** `flair>=0.13`, `sumy` (TextRank), `transformers` (XNLI zero-shot), `pytest`, `unittest.mock`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `tests/conftest.py` | Create | Shared pytest fixtures |
| `tests/test_extractor.py` | Create | TextRank extraction unit tests |
| `tests/test_nli.py` | Create | NLI classify() unit tests |
| `tests/test_ner.py` | Create | Flair NER unit tests |
| `tests/test_geotagger_service.py` | Create | Scope/city NLI + street rescue tests |
| `tests/test_classifier_service.py` | Create | geo_scope passthrough tests |
| `nlp/extractor/extractive.py` | Create (move) | TextRank sentence extractor (from summarizer/) |
| `nlp/extractor/service.py` | Patch | Swap TF-IDF for TextRank |
| `nlp/nli.py` | Create | Shared XNLI pipeline singleton |
| `nlp/classifier/model.py` | Simplify | Thin import shim over nlp.nli |
| `nlp/classifier/service.py` | Patch | Accept geo_scope; skip scope NLI when provided |
| `nlp/geotagger/ner.py` | Rewrite | Flair SequenceTagger replaces HF pipeline |
| `nlp/geotagger/service.py` | Rewrite | Inline NLI cascade + street rescue; delete disambiguator logic |
| `nlp/geotagger/disambiguator.py` | Delete | Replaced by service.py |
| `nlp/summarizer/extractive.py` | Delete | Moved to nlp/extractor/extractive.py |
| `api/models.py` | Patch | Add `geo_scope: str | None = None` to ClassifyRequest |
| `api/main.py` | Patch | Flair device init + use nli._ensure_loaded() |
| `requirements.txt` | Update | Add flair>=0.13; remove scikit-learn |
| `Dockerfile` | Update | Swap NER model pre-pull |
| `eval/06_e2e_pipeline.ipynb` | Update | Pass geo_scope from /geotag into /classify |

---

## Task 1: Test infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `pytest.ini`

- [ ] **Step 1: Create pytest.ini**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

- [ ] **Step 2: Create tests/__init__.py and conftest.py**

`tests/__init__.py` — empty file.

`tests/conftest.py`:
```python
import pytest


@pytest.fixture
def madrid_text():
    return (
        "El Ayuntamiento de Madrid ha aprobado la ampliación del carril bici "
        "en la Gran Vía. Las obras comenzarán en junio con un presupuesto de "
        "2,4 millones de euros."
    )


@pytest.fixture
def barcelona_text():
    return (
        "Barcelona supera a Madrid en kilómetros de carril bici según un estudio "
        "reciente. La capital catalana lidera el ranking nacional de movilidad "
        "sostenible, seguida de cerca por Sevilla."
    )


@pytest.fixture
def national_text():
    return (
        "El Ministerio de Transportes ha presentado la nueva Estrategia Nacional "
        "de Movilidad Ciclista 2026-2030 ante el Congreso. El plan dotará con "
        "2.000 millones a municipios de todo España."
    )


@pytest.fixture
def regional_text():
    return (
        "La Comunidad de Madrid ha aprobado un plan para conectar los municipios "
        "del corredor del Henares mediante una red ciclista interurbana de 120 km. "
        "La inversión beneficiará a Alcalá de Henares, Torrejón de Ardoz y Coslada."
    )
```

- [ ] **Step 3: Verify pytest can be found**

```bash
cd /path/to/nlp_service && python -m pytest --collect-only
```
Expected: `no tests ran` (no test files yet — that's fine)

- [ ] **Step 4: Commit**

```bash
git add pytest.ini tests/__init__.py tests/conftest.py
git commit -m "test: add pytest infrastructure and shared fixtures"
```

---

## Task 2: Move extractive.py + wire TextRank in extractor

**Files:**
- Create: `nlp/extractor/extractive.py` (copy of `nlp/summarizer/extractive.py`)
- Modify: `nlp/extractor/service.py`
- Create: `tests/test_extractor.py`

- [ ] **Step 1: Write failing tests**

`tests/test_extractor.py`:
```python
import pytest
from nlp.extractor.service import extract_and_embed, _textrank_extract, _split_sentences


def test_split_sentences_basic():
    text = "Primera frase. Segunda frase. Tercera frase."
    sentences = _split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "Primera frase."


def test_short_text_returned_unchanged():
    text = "Solo una frase."
    result = _textrank_extract(text, max_words=200)
    assert result == text


def test_two_sentence_text_returned_unchanged():
    text = "Primera frase. Segunda frase."
    result = _textrank_extract(text, max_words=200)
    assert result == text


def test_textrank_respects_max_words():
    # Build a long text with 10 distinct sentences
    sentences = [f"Esta es la frase número {i} sobre ciclismo en la ciudad." for i in range(10)]
    text = " ".join(sentences)
    result = _textrank_extract(text, max_words=30)
    word_count = len(result.split())
    assert word_count <= 40  # allow slight overshoot from sentence boundaries


def test_textrank_returns_non_empty_for_long_text(madrid_text):
    result = _textrank_extract(madrid_text, max_words=200)
    assert len(result.strip()) > 0


def test_extract_and_embed_returns_correct_shapes(madrid_text):
    extract, embedding = extract_and_embed(madrid_text)
    assert isinstance(extract, str)
    assert len(extract) > 0
    assert embedding.shape == (384,)
    assert embedding.dtype.name == "float32"
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python -m pytest tests/test_extractor.py -v
```
Expected: `ImportError` or `AttributeError` — `_textrank_extract` does not exist yet.

- [ ] **Step 3: Create nlp/extractor/extractive.py**

Copy `nlp/summarizer/extractive.py` to `nlp/extractor/extractive.py` — identical content:

```python
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

_LANGUAGE = "spanish"
_summarizer = None


def _ensure_loaded() -> None:
    global _summarizer
    if _summarizer is None:
        _summarizer = TextRankSummarizer(Stemmer(_LANGUAGE))
        _summarizer.stop_words = get_stop_words(_LANGUAGE)


def extract_top_sentences(text: str, n: int) -> str:
    _ensure_loaded()
    tokenizer = Tokenizer(_LANGUAGE)
    parser = PlaintextParser.from_string(text, tokenizer)
    sentences = _summarizer(parser.document, n)
    return " ".join(str(s) for s in sentences)
```

- [ ] **Step 4: Rewrite nlp/extractor/service.py**

```python
import re
import numpy as np
from nlp.encoder import encode
from nlp.extractor.extractive import extract_top_sentences


def extract_and_embed(text: str, max_words: int = 200) -> tuple[str, np.ndarray]:
    extract = _textrank_extract(text, max_words)
    return extract, encode(extract)


def _textrank_extract(text: str, max_words: int) -> str:
    sentences = _split_sentences(text)
    if len(sentences) <= 2:
        return text
    avg_words = sum(len(s.split()) for s in sentences) / len(sentences)
    n = max(1, int(max_words / max(avg_words, 1)))
    return extract_top_sentences(text, n)


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
python -m pytest tests/test_extractor.py -v
```
Expected: all 6 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add nlp/extractor/extractive.py nlp/extractor/service.py tests/test_extractor.py
git commit -m "feat: replace TF-IDF with TextRank in extractor; wire up extractive.py"
```

---

## Task 3: Shared NLI module + simplify classifier/model.py

**Files:**
- Create: `nlp/nli.py`
- Modify: `nlp/classifier/model.py`
- Create: `tests/test_nli.py`

- [ ] **Step 1: Write failing tests**

`tests/test_nli.py`:
```python
from unittest.mock import MagicMock, patch
import pytest


def _make_mock_pipeline(labels, scores):
    mock = MagicMock()
    mock.return_value = {"labels": labels, "scores": scores, "sequence": "test"}
    return mock


def test_classify_returns_expected_keys():
    with patch("nlp.nli._pipeline", _make_mock_pipeline(["a", "b"], [0.8, 0.2])):
        from nlp import nli
        result = nli.classify("some text", labels=["a", "b"], multi_label=False)
    assert "labels" in result
    assert "scores" in result


def test_classify_passes_hypothesis_template():
    mock_pipe = _make_mock_pipeline(["Madrid", "Barcelona"], [0.7, 0.3])
    with patch("nlp.nli._pipeline", mock_pipe):
        from nlp import nli
        nli.classify(
            "El ayuntamiento trabaja en Madrid.",
            labels=["Madrid", "Barcelona"],
            multi_label=False,
            hypothesis_template="Este artículo trata sobre {}.",
        )
    call_kwargs = mock_pipe.call_args[1]
    assert call_kwargs["hypothesis_template"] == "Este artículo trata sobre {}."


def test_classify_multi_label_flag_passed():
    mock_pipe = _make_mock_pipeline(["topic"], [0.9])
    with patch("nlp.nli._pipeline", mock_pipe):
        from nlp import nli
        nli.classify("text", labels=["topic"], multi_label=True)
    assert mock_pipe.call_args[1]["multi_label"] is True


def test_classifier_model_still_works_via_shim():
    mock_pipe = _make_mock_pipeline(["label"], [0.6])
    with patch("nlp.nli._pipeline", mock_pipe):
        from nlp.classifier import model
        result = model.classify("text", labels=["label"], multi_label=False)
    assert result["labels"] == ["label"]
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python -m pytest tests/test_nli.py -v
```
Expected: `ImportError` — `nlp.nli` does not exist.

- [ ] **Step 3: Create nlp/nli.py**

```python
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
```

- [ ] **Step 4: Simplify nlp/classifier/model.py**

```python
from nlp.nli import _ensure_loaded, classify  # noqa: F401  re-export for backward compat
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
python -m pytest tests/test_nli.py -v
```
Expected: all 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add nlp/nli.py nlp/classifier/model.py tests/test_nli.py
git commit -m "feat: extract shared NLI module; simplify classifier/model.py to import shim"
```

---

## Task 4: Rewrite nlp/geotagger/ner.py (Flair)

**Files:**
- Modify: `nlp/geotagger/ner.py`
- Create: `tests/test_ner.py`

- [ ] **Step 1: Write failing tests**

`tests/test_ner.py`:
```python
from unittest.mock import MagicMock, patch
import pytest
from nlp.geotagger.ner import Span


def _make_flair_entity(text, tag, start, end):
    e = MagicMock()
    e.text = text
    e.tag = tag
    e.start_position = start
    e.end_position = end
    return e


def _make_mock_tagger(entities):
    tagger = MagicMock()

    def predict_side_effect(sentence):
        sentence.get_spans.return_value = entities

    tagger.predict.side_effect = predict_side_effect
    return tagger


def test_extract_spans_returns_only_loc(madrid_text):
    entities = [
        _make_flair_entity("Madrid", "LOC", 19, 25),
        _make_flair_entity("Juan García", "PER", 50, 61),
    ]
    mock_tagger = _make_mock_tagger(entities)

    with patch("nlp.geotagger.ner._tagger", mock_tagger):
        from nlp.geotagger.ner import extract_spans
        spans = extract_spans(madrid_text)

    assert len(spans) == 1
    assert spans[0].text == "Madrid"
    assert spans[0].label == "LOC"


def test_extract_spans_includes_street_regex_hits():
    text = "El corte afecta a la Calle Alcalá entre Goya y Velázquez."
    entities = []  # Flair finds nothing
    mock_tagger = _make_mock_tagger(entities)

    with patch("nlp.geotagger.ner._tagger", mock_tagger):
        from nlp.geotagger.ner import extract_spans
        spans = extract_spans(text)

    street_spans = [s for s in spans if s.hint == "street"]
    assert len(street_spans) >= 1
    assert any("Calle Alcalá" in s.text for s in street_spans)


def test_extract_spans_no_subword_artifacts():
    # Flair returns clean words — confirm no ## in span text
    entities = [
        _make_flair_entity("Eixample", "LOC", 15, 23),
    ]
    mock_tagger = _make_mock_tagger(entities)

    with patch("nlp.geotagger.ner._tagger", mock_tagger):
        from nlp.geotagger.ner import extract_spans
        spans = extract_spans("Los vecinos del Eixample se quejan.")

    assert all("##" not in s.text for s in spans)
    assert spans[0].text == "Eixample"


def test_span_dataclass_fields():
    span = Span(text="Madrid", label="LOC", start_char=0, end_char=6)
    assert span.text == "Madrid"
    assert span.label == "LOC"
    assert span.hint == ""
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python -m pytest tests/test_ner.py -v
```
Expected: tests fail — Flair not imported yet.

- [ ] **Step 3: Rewrite nlp/geotagger/ner.py**

```python
import os
import re
from dataclasses import dataclass, field

import torch
import flair
from flair.models import SequenceTagger
from flair.data import Sentence

flair.device = torch.device(os.environ.get("NER_DEVICE", "cpu"))

_KEEP_LABELS = {"LOC"}
_tagger: SequenceTagger | None = None

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
        _tagger = SequenceTagger.load("flair/ner-spanish-large")


def extract_spans(text: str) -> list[Span]:
    _ensure_loaded()
    assert _tagger is not None

    sentence = Sentence(text, use_tokenizer=True)
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
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
python -m pytest tests/test_ner.py -v
```
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add nlp/geotagger/ner.py tests/test_ner.py
git commit -m "feat: rewrite geotagger NER with Flair ner-spanish-large"
```

---

## Task 5: Rewrite nlp/geotagger/service.py (NLI cascade + street rescue)

**Files:**
- Modify: `nlp/geotagger/service.py`
- Create: `tests/test_geotagger_service.py`

- [ ] **Step 1: Write failing tests**

`tests/test_geotagger_service.py`:
```python
from unittest.mock import MagicMock, patch
import pytest


def _nli_scope_result(winner: str):
    """Return a mock nli.classify result where winner scores 0.9."""
    h_local = (
        "Este artículo describe actuaciones, obras o iniciativas "
        "de un ayuntamiento o municipio concreto de España."
    )
    h_regional = (
        "Este artículo describe políticas o actuaciones de una comunidad autónoma, "
        "diputación provincial o región de España que afectan a varios municipios."
    )
    h_national = (
        "Este artículo describe una política, ley, normativa o acontecimiento "
        "de alcance estatal en España, sin limitarse a una ciudad o región concreta."
    )
    scores = {h_local: 0.1, h_regional: 0.1, h_national: 0.1}
    winner_map = {"local": h_local, "regional": h_regional, "national": h_national}
    scores[winner_map[winner]] = 0.9
    labels = sorted(scores, key=lambda k: scores[k], reverse=True)
    return {"labels": labels, "scores": [scores[l] for l in labels], "sequence": ""}


def _nli_city_result(winner_city: str, all_cities: list[str]):
    scores = {c: 0.1 for c in all_cities}
    scores[winner_city] = 0.9
    labels = sorted(scores, key=lambda k: scores[k], reverse=True)
    return {"labels": labels, "scores": [scores[l] for l in labels], "sequence": ""}


def test_scope_national_returns_no_city(national_text):
    with patch("nlp.geotagger.service.ner") as mock_ner, \
         patch("nlp.geotagger.service.gazetteer") as mock_geo, \
         patch("nlp.nli.classify") as mock_nli:

        mock_ner.extract_spans.return_value = []
        mock_geo.lookup.return_value = []
        mock_geo.lookup_street.return_value = []
        mock_geo.lookup_street_all_cities.return_value = {}
        mock_geo.get_city_prior.return_value = None
        mock_geo._STREET_PREFIX_RE.match.return_value = None
        mock_nli.return_value = _nli_scope_result("national")

        from nlp.geotagger import service
        service._cities = []  # empty cities DB
        result = service.run(national_text, headline="", source="")

    assert result["geo_scope"] == "national"
    assert result["geo_cities"] == []


def test_scope_local_single_city_no_stage2(madrid_text):
    from nlp.geotagger.gazetteer import GeoEntry

    madrid_entry = GeoEntry(
        geonames_id=3117735, name="Madrid", lat=40.4165, lon=-3.7026,
        feature_class="P", feature_code="PPLC", admin1_code="29", population=3200000,
    )

    with patch("nlp.geotagger.service.ner") as mock_ner, \
         patch("nlp.geotagger.service.gazetteer") as mock_geo, \
         patch("nlp.nli.classify") as mock_nli:

        mock_ner.extract_spans.return_value = []
        mock_geo.lookup.side_effect = lambda span: [madrid_entry] if "Madrid" in span else []
        mock_geo.lookup_street.return_value = []
        mock_geo.lookup_street_all_cities.return_value = {}
        mock_geo.get_city_prior.return_value = None
        mock_geo._STREET_PREFIX_RE.match.return_value = None
        mock_nli.return_value = _nli_scope_result("local")

        from nlp.geotagger import service
        service._cities = [{"id": 3117735, "name": "Madrid", "population": 3200000}]
        service._by_name = {"madrid": {"id": 3117735, "name": "Madrid", "population": 3200000}}
        result = service.run(madrid_text, headline="Carril bici en Madrid", source="")

    assert result["geo_scope"] == "city"
    # Stage 2 NLI should NOT be called for single-candidate case
    assert mock_nli.call_count == 1


def test_street_rescue_routes_to_geo_streets():
    from nlp.geotagger.gazetteer import GeoEntry
    from nlp.geotagger.ner import Span

    calle_span = Span(text="Calle Alcalá", label="LOC", start_char=10, end_char=22)

    with patch("nlp.geotagger.service.ner") as mock_ner, \
         patch("nlp.geotagger.service.gazetteer") as mock_geo, \
         patch("nlp.nli.classify") as mock_nli:

        mock_ner.extract_spans.return_value = [calle_span]
        # Gazetteer returns nothing for "Calle Alcalá"
        mock_geo.lookup.return_value = []
        # Street prefix regex matches
        import re
        mock_geo._STREET_PREFIX_RE = re.compile(
            r'^(?:calle|avda?\.?|avenida|plaza)\s+', re.IGNORECASE
        )
        mock_geo.lookup_street.return_value = [101, 102]
        mock_geo.lookup_street_all_cities.return_value = {}
        mock_geo.get_city_prior.return_value = None
        mock_nli.return_value = _nli_scope_result("local")

        from nlp.geotagger import service
        service._cities = []
        service._by_name = {}
        result = service.run(
            "El corte afecta a la Calle Alcalá.",
            headline="",
            source="",
        )

    # Rescued span must appear in all_places as "street" type
    street_places = [p for p in result["all_places"] if p["type"] == "street"]
    assert len(street_places) >= 1
    assert any("Alcalá" in p["text"] for p in street_places)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python -m pytest tests/test_geotagger_service.py -v
```
Expected: failures — service.py still has old logic.

- [ ] **Step 3: Rewrite nlp/geotagger/service.py**

```python
from __future__ import annotations

import json
import logging
import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import gazetteer, ner
from nlp import nli

log = logging.getLogger("nlp_service.geotagger")

_CITIES_PATH = Path(os.environ.get(
    "CITIES_SNAPSHOT_PATH",
    Path(__file__).parent.parent.parent / "nlp" / "geotagger" / "data" / "cities_snapshot.json",
))
_cities: list[dict] = []
_by_name: dict[str, dict] = {}
_max_population = 1

# ── NLI hypothesis constants ──────────────────────────────────────────────────
_H_LOCAL = (
    "Este artículo describe actuaciones, obras o iniciativas "
    "de un ayuntamiento o municipio concreto de España."
)
_H_REGIONAL = (
    "Este artículo describe políticas o actuaciones de una comunidad autónoma, "
    "diputación provincial o región de España que afectan a varios municipios."
)
_H_NATIONAL = (
    "Este artículo describe una política, ley, normativa o acontecimiento "
    "de alcance estatal en España, sin limitarse a una ciudad o región concreta."
)
_SCOPE_THRESHOLD = 0.35
_MAX_CITY_CANDIDATES = 10
_CITY_TEMPLATE = "Este artículo describe principalmente hechos o iniciativas en {}."
_REGION_TEMPLATE = (
    "Este artículo trata principalmente sobre "
    "la comunidad autónoma, provincia o región de {}."
)


@dataclass
class CityHit:
    city_name: str
    city_id: int
    lat: float
    lon: float
    geonames_id: int | None
    score: float


def _normalize(s: str) -> str:
    s = s.lower()
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def _load_cities() -> None:
    global _cities, _by_name, _max_population
    if _cities:
        return
    if not _CITIES_PATH.exists():
        return
    _cities = json.loads(_CITIES_PATH.read_text(encoding="utf-8"))
    for c in _cities:
        _by_name[_normalize(c["name"])] = c
        if c.get("alt_name"):
            _by_name[_normalize(c["alt_name"])] = c
    if _cities:
        _max_population = max(c.get("population") or 1 for c in _cities)


def _match_city(geo_entry: gazetteer.GeoEntry) -> dict | None:
    return _by_name.get(_normalize(geo_entry.name))


def load() -> None:
    gazetteer.load()
    gazetteer.load_streets()
    gazetteer.load_source_prior()
    _load_cities()


def _classify_geo(
    spans_with_geo: list[tuple[str, list[gazetteer.GeoEntry]]],
    premise: str,
) -> tuple[str, CityHit | None, str | None]:
    """Run 2-level NLI to determine scope, winning city, and winning region.

    Returns (geo_scope, city_hit, geo_region).
    geo_scope is one of 'national', 'regional', 'city'.
    """
    # Build candidate pools — deduplicate by city_id / entry name
    city_pool: list[tuple[gazetteer.GeoEntry, dict]] = []
    region_pool: list[gazetteer.GeoEntry] = []
    seen_city_ids: set[int] = set()
    seen_region_names: set[str] = set()

    for _span_text, entries in spans_with_geo:
        if not entries:
            continue
        best = max(entries, key=lambda e: e.population)
        if best.feature_class == "P":
            city = _match_city(best)
            if city and city["id"] not in seen_city_ids:
                city_pool.append((best, city))
                seen_city_ids.add(city["id"])
        elif best.feature_class == "A":
            if best.name not in seen_region_names:
                region_pool.append(best)
                seen_region_names.add(best.name)

    # Stage 1: scope
    scope_result = nli.classify(
        premise,
        labels=[_H_LOCAL, _H_REGIONAL, _H_NATIONAL],
        multi_label=False,
    )
    score_map = dict(zip(scope_result["labels"], scope_result["scores"]))
    local_s = score_map.get(_H_LOCAL, 0.0)
    regional_s = score_map.get(_H_REGIONAL, 0.0)
    national_s = score_map.get(_H_NATIONAL, 0.0)
    best_score = max(local_s, regional_s, national_s)

    if best_score < _SCOPE_THRESHOLD:
        # Heuristic fallback
        geo_scope = "city" if city_pool else ("regional" if region_pool else "national")
    elif local_s >= regional_s and local_s >= national_s:
        geo_scope = "city"
    elif regional_s >= local_s and regional_s >= national_s:
        geo_scope = "regional"
    else:
        geo_scope = "national"

    city_hit: CityHit | None = None
    geo_region: str | None = None

    if geo_scope == "city":
        city_hit = _pick_city(city_pool, premise)
    elif geo_scope == "regional":
        geo_region = _pick_region(region_pool, premise)

    return geo_scope, city_hit, geo_region


def _pick_city(
    city_pool: list[tuple[gazetteer.GeoEntry, dict]],
    premise: str,
) -> CityHit | None:
    if not city_pool:
        return None
    if len(city_pool) == 1:
        entry, city = city_pool[0]
        return CityHit(
            city_name=city["name"], city_id=city["id"],
            lat=entry.lat, lon=entry.lon,
            geonames_id=entry.geonames_id, score=1.0,
        )
    # Limit to top-10 by population to keep label set small
    candidates = sorted(city_pool, key=lambda x: x[0].population, reverse=True)
    candidates = candidates[:_MAX_CITY_CANDIDATES]
    city_names = [city["name"] for _, city in candidates]

    result = nli.classify(premise, labels=city_names, multi_label=False,
                          hypothesis_template=_CITY_TEMPLATE)
    score_map = dict(zip(result["labels"], result["scores"]))
    best_name = result["labels"][0]
    best_score = result["scores"][0]

    matched = next(
        ((e, c) for e, c in candidates if c["name"] == best_name), None
    )
    if matched is None:
        return None
    entry, city = matched
    return CityHit(
        city_name=city["name"], city_id=city["id"],
        lat=entry.lat, lon=entry.lon,
        geonames_id=entry.geonames_id, score=best_score,
    )


def _pick_region(region_pool: list[gazetteer.GeoEntry], premise: str) -> str | None:
    if not region_pool:
        return None
    if len(region_pool) == 1:
        return region_pool[0].name
    region_names = [e.name for e in region_pool]
    result = nli.classify(premise, labels=region_names, multi_label=False,
                          hypothesis_template=_REGION_TEMPLATE)
    return result["labels"][0]


def run(
    text: str,
    headline: str = "",
    source: str = "",
) -> dict[str, Any]:
    load()

    full_input = f"{headline}. {text}" if headline else text
    premise = full_input

    # Stage A: NER
    spans = ner.extract_spans(full_input)
    street_spans = [s for s in spans if s.hint == "street"]
    loc_spans = [s for s in spans if s.hint != "street"]

    log.info("Stage A — loc=%s street=%s",
             [s.text for s in loc_spans], [s.text for s in street_spans])

    # Stage B1: Gazetteer lookup
    source_prior_city_id = gazetteer.get_city_prior(source) if source else None
    spans_with_geo = [(s.text, gazetteer.lookup(s.text)) for s in loc_spans]

    log.info("Stage B1 — gazetteer hits: %s",
             {t: [e.name for e in entries] for t, entries in spans_with_geo})

    # Rescue LOC spans the gazetteer missed but that carry a street prefix
    rescued: list[ner.Span] = [
        s for s, entries in zip(loc_spans, [e for _, e in spans_with_geo])
        if not entries and gazetteer._STREET_PREFIX_RE.match(s.text)
    ]
    for r in rescued:
        loc_spans = [s for s in loc_spans if s is not r]
        spans_with_geo = [(t, e) for t, e in spans_with_geo if t != r.text or e]
        street_spans.append(ner.Span(
            text=r.text, label=r.label,
            start_char=r.start_char, end_char=r.end_char,
            hint="street",
        ))

    # Stage B2: NLI geo-classification
    geo_scope, city_hit, geo_region = _classify_geo(spans_with_geo, premise)

    log.info("Stage B2 — scope=%s city=%s region=%s",
             geo_scope,
             city_hit.city_name if city_hit else None,
             geo_region)

    winning_city_id = city_hit.city_id if city_hit else None

    # Stage B3: Street resolution
    geo_streets: list[dict] = []
    for s in street_spans:
        if winning_city_id:
            edge_ids = gazetteer.lookup_street(winning_city_id, s.text)
            if edge_ids:
                geo_streets.append({"span": s.text, "edge_ids": edge_ids,
                                    "city_id": winning_city_id})
                continue
        matches = gazetteer.lookup_street_all_cities(s.text)
        if len(matches) == 1:
            cid, edge_ids = next(iter(matches.items()))
            geo_streets.append({"span": s.text, "edge_ids": edge_ids, "city_id": cid})
        elif len(matches) > 1 and source_prior_city_id and source_prior_city_id in matches:
            geo_streets.append({"span": s.text,
                                 "edge_ids": matches[source_prior_city_id],
                                 "city_id": source_prior_city_id})
        else:
            geo_streets.append({"span": s.text, "edge_ids": [], "city_id": None})

    # Build geo_points from P-class entries when no city was resolved
    geo_points: list[dict] = []
    for _span_text, entries in spans_with_geo:
        for entry in entries:
            if entry.feature_class == "P" and not city_hit:
                best = max(entries, key=lambda x: x.population)
                geo_points.append({
                    "span": _span_text, "lat": best.lat, "lon": best.lon,
                    "geonames_id": best.geonames_id,
                })
                break

    # all_places backward-compat
    geo_cities: list[dict] = []
    if city_hit:
        geo_cities = [{
            "city_id": city_hit.city_id,
            "city_name": city_hit.city_name,
            "confidence": city_hit.score,
        }]

    all_places: list[dict] = []
    for span_text, entries in spans_with_geo:
        if not entries:
            all_places.append({"text": span_text, "type": "other",
                                "lat": None, "lon": None,
                                "geonames_id": None, "city_id": None})
            continue
        best = max(entries, key=lambda x: x.population)
        ptype = ("city" if best.feature_class == "P"
                 else ("region" if best.feature_class == "A" else "other"))
        cid = city_hit.city_id if (city_hit and best.geonames_id == city_hit.geonames_id) else None
        all_places.append({
            "text": span_text, "type": ptype,
            "lat": best.lat, "lon": best.lon,
            "geonames_id": best.geonames_id, "city_id": cid,
        })
    for s in street_spans:
        all_places.append({"text": s.text, "type": "street",
                            "lat": None, "lon": None,
                            "geonames_id": None, "city_id": winning_city_id})

    log.info("Stage B3 — streets=%s points=%s",
             [s["span"] for s in geo_streets], [p["span"] for p in geo_points])

    return {
        "geo_scope": geo_scope,
        "geo_region": geo_region,
        "geo_cities": geo_cities,
        "geo_streets": geo_streets,
        "geo_points": geo_points,
        "all_places": all_places,
        "city": city_hit.city_name if city_hit else None,
        "city_confidence": city_hit.score if city_hit else 0.0,
    }
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
python -m pytest tests/test_geotagger_service.py -v
```
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add nlp/geotagger/service.py tests/test_geotagger_service.py
git commit -m "feat: rewrite geotagger service with 2-level NLI cascade and street rescue"
```

---

## Task 6: Patch ClassifyRequest + classifier/service.py

**Files:**
- Modify: `api/models.py`
- Modify: `nlp/classifier/service.py`
- Create: `tests/test_classifier_service.py`

- [ ] **Step 1: Write failing tests**

`tests/test_classifier_service.py`:
```python
from unittest.mock import MagicMock, patch
import pytest


def _mock_taxonomy():
    tax = MagicMock()
    tax.relevance_hypothesis = ""
    tax.relevance_threshold = 0.4
    tax.blacklist_labels = []
    tax.blacklist_threshold = 0.7
    tax.labels = ["carril bici"]
    tax.score_threshold = 0.5
    tax.top_k = 3
    tax.scope_hypotheses = {"city": "hip city", "national": "hip national"}
    tax.scope_threshold = 0.35
    return tax


def test_geo_scope_provided_skips_scope_nli(madrid_text):
    topic_result = {"labels": ["carril bici"], "scores": [0.8], "sequence": ""}
    with patch("nlp.classifier.service.taxonomy") as mock_tax, \
         patch("nlp.nli.classify") as mock_nli:
        mock_tax.load.return_value = _mock_taxonomy()
        mock_nli.return_value = topic_result

        from nlp.classifier.service import run
        result = run(
            summary=madrid_text,
            geo_cities=[],
            geo_scope="city",  # provided by geotagger
        )

    # nli.classify should only be called once (topics), not twice (topics + scope)
    assert mock_nli.call_count == 1
    assert result["geo_scope"] == "city"


def test_geo_scope_absent_runs_scope_nli(madrid_text):
    topic_result = {"labels": ["carril bici"], "scores": [0.8], "sequence": ""}
    scope_result = {"labels": ["hip city"], "scores": [0.9], "sequence": ""}
    call_results = [topic_result, scope_result]

    with patch("nlp.classifier.service.taxonomy") as mock_tax, \
         patch("nlp.nli.classify", side_effect=call_results) as mock_nli:
        mock_tax.load.return_value = _mock_taxonomy()

        from nlp.classifier.service import run
        result = run(summary=madrid_text, geo_cities=[], geo_scope=None)

    assert mock_nli.call_count == 2  # topics + scope
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python -m pytest tests/test_classifier_service.py -v
```
Expected: `TypeError` — `run()` does not accept `geo_scope` yet.

- [ ] **Step 3: Patch api/models.py — add geo_scope to ClassifyRequest**

Find the `ClassifyRequest` class (line ~92) and add the field:

```python
class ClassifyRequest(BaseModel):
    article_id: str
    summary: str
    geo_cities: list[GeoCity] = []
    search_tags: list[str] = []
    source_profile: SourceProfile | None = None
    geo_scope: str | None = None
```

- [ ] **Step 4: Patch nlp/classifier/service.py — accept and use geo_scope**

Change the `run()` signature and scope resolution:

```python
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

    if tax.relevance_hypothesis:
        rel = model.classify(summary, labels=[tax.relevance_hypothesis], multi_label=True)
        rel_score = rel["scores"][0] if rel["scores"] else 0.0
        if rel_score < tax.relevance_threshold:
            return {"topics": [], "scores": {}, "geo_scope": geo_scope or "national",
                    "out_of_scope": True}

    tag_prefix = ""
    if search_tags:
        tag_prefix = "Artículo buscado por: " + ", ".join(f"'{t}'" for t in search_tags) + ". "
    topic_premise = tag_prefix + summary

    all_labels = tax.labels + tax.blacklist_labels
    raw = model.classify(topic_premise, labels=all_labels, multi_label=True)
    scored = dict(zip(raw["labels"], raw["scores"]))

    if tax.blacklist_labels:
        top_blacklist_score = max(scored.get(lbl, 0.0) for lbl in tax.blacklist_labels)
        if top_blacklist_score >= tax.blacklist_threshold:
            return {"topics": [], "scores": scored,
                    "geo_scope": geo_scope or "national", "out_of_scope": True}

    filtered = sorted(
        [(lbl, scored[lbl]) for lbl in tax.labels if scored.get(lbl, 0) >= tax.score_threshold],
        key=lambda x: x[1], reverse=True,
    )[:tax.top_k]

    # Use provided geo_scope (from geotagger) or run own NLI scope pass
    resolved_scope = geo_scope or _scope_pass(summary, geo_cities, source_profile, tax)

    return {
        "topics": [lbl for lbl, _ in filtered],
        "scores": {lbl: scored[lbl] for lbl in tax.labels},
        "geo_scope": resolved_scope,
        "out_of_scope": False,
    }
```

Also update the classify router to pass `geo_scope` from the request:

In `api/routers/classify.py`, change:
```python
result = classifier_service.run(
    summary=req.summary,
    geo_cities=[c.model_dump() for c in req.geo_cities],
    search_tags=req.search_tags,
    source_profile=req.source_profile.model_dump() if req.source_profile else None,
    geo_scope=req.geo_scope,
)
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
python -m pytest tests/test_classifier_service.py -v
```
Expected: both tests PASS.

- [ ] **Step 6: Commit**

```bash
git add api/models.py nlp/classifier/service.py api/routers/classify.py tests/test_classifier_service.py
git commit -m "feat: add geo_scope passthrough to ClassifyRequest; skip scope NLI when provided"
```

---

## Task 7: Update api/main.py (Flair device init + nli warmup)

**Files:**
- Modify: `api/main.py`

- [ ] **Step 1: Update the lifespan function**

In `api/main.py`, replace the lifespan imports and warmup sequence:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("nlp-service starting up")

    import os
    import torch
    import flair as _flair_lib

    # Set Flair device before any model loads
    _flair_lib.device = torch.device(os.environ.get("NER_DEVICE", "cpu"))

    from nlp import nli as _nli
    from nlp.encoder import load_encoder as _load_encoder
    from nlp.geotagger import ner as _ner
    from nlp.geotagger import service as _geo_svc
    from nlp.classifier import service as _cls_svc
    from nlp.dedup import service as _dedup_svc
    from nlp.dedup import embedding_index as _emb_idx
    from api.warmth import mark_warm as _mark_warm

    _load_encoder()
    _nli._ensure_loaded()       # shared by classifier + geotagger
    _ner._ensure_loaded()
    _geo_svc.load()
    _cls_svc.load()
    _dedup_svc.load()
    _emb_idx._ensure_loaded()

    _mark_warm("extract")
    _mark_warm("geotag")
    _mark_warm("classify")
    _mark_warm("dedup")
    _mark_warm("summarize")
    log.info("nlp-service ready")
    yield
    log.info("nlp-service shutting down — flushing dedup state")
    try:
        from nlp.dedup import service as dedup_service
        dedup_service.flush()
    except Exception:
        log.exception("dedup flush on shutdown failed (continuing)")
```

- [ ] **Step 2: Verify import chain is clean**

```bash
python -c "from api.main import app; print('import ok')"
```
Expected: `import ok`

- [ ] **Step 3: Commit**

```bash
git add api/main.py
git commit -m "feat: init Flair device in lifespan; use shared nli._ensure_loaded()"
```

---

## Task 8: Delete dead code + update requirements.txt

**Files:**
- Delete: `nlp/geotagger/disambiguator.py`
- Delete: `nlp/summarizer/extractive.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Delete dead files**

```bash
git rm nlp/geotagger/disambiguator.py nlp/summarizer/extractive.py
```

- [ ] **Step 2: Update requirements.txt**

Remove `scikit-learn==1.5.2`. Add `flair>=0.13`. Result:

```
fastapi==0.115.0
uvicorn[standard]==0.32.0
httpx==0.27.2
pydantic==2.9.2
PyYAML==6.0.2
transformers==4.45.2
torch>=2.5.0
sentence-transformers==3.2.1
datasketch==1.6.5
faiss-cpu>=1.9.0
sumy==0.11.0
nltk==3.9.1
numpy==1.26.4
flair>=0.13
```

- [ ] **Step 3: Verify nothing imports the deleted files**

```bash
grep -r "from nlp.geotagger.disambiguator\|from nlp.geotagger import disambiguator\|from nlp.summarizer.extractive\|from nlp.summarizer import extractive" --include="*.py" .
```
Expected: no output.

- [ ] **Step 4: Run full test suite**

```bash
python -m pytest tests/ -v
```
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt
git commit -m "chore: delete disambiguator.py and summarizer/extractive.py; remove scikit-learn; add flair"
```

---

## Task 9: Update Dockerfile

**Files:**
- Modify: `Dockerfile`

- [ ] **Step 1: Swap NER model pre-pull**

Replace the `mrm8488` NER pre-pull line:
```dockerfile
RUN python -c "from transformers import pipeline; pipeline('token-classification', model='mrm8488/bert-spanish-cased-finetuned-ner', aggregation_strategy='simple')"
```

With:
```dockerfile
# NER: Flair Spanish large (XLM-R + character LM, F1=90.54)
RUN python -c "from flair.models import SequenceTagger; SequenceTagger.load('flair/ner-spanish-large')"
```

- [ ] **Step 2: Verify Dockerfile builds (dry-run syntax check)**

```bash
docker build --no-cache --target base . 2>&1 | head -20
```
Or simply check the file looks correct:
```bash
grep -n "flair\|mrm8488" Dockerfile
```
Expected: only `flair/ner-spanish-large` appears; no `mrm8488`.

- [ ] **Step 3: Commit**

```bash
git add Dockerfile
git commit -m "feat: update Dockerfile to pre-pull flair/ner-spanish-large"
```

---

## Task 10: Update E2E notebook to pass geo_scope

**Files:**
- Modify: `eval/06_e2e_pipeline.ipynb`

- [ ] **Step 1: Update the classify cell**

In `eval/06_e2e_pipeline.ipynb`, find the classify cell (`cell-classify`) and update the JSON body to include `geo_scope` from the geotag result:

```python
# ── Step 5: /classify ────────────────────────────────────────────────────────
t0 = time.monotonic()
resp = requests.post(
    f'{NLP_BASE_URL}/classify',
    json={
        'article_id': ARTICLE['article_id'],
        'summary':    sum_result['summary'],
        'geo_cities': geo_result['geo_cities'],
        'search_tags': [],
        'source_profile': None,
        'geo_scope':  geo_result['geo_scope'],   # ← geotagger is authoritative
    },
    headers=HEADERS,
)
print(f'/classify  {resp.status_code}  {time.monotonic()-t0:.2f}s')
assert resp.status_code == 200, resp.text
cls_result = resp.json()

print(f'  topics={cls_result["topics"]}')
print(f'  geo_scope={cls_result["geo_scope"]!r}  (from geotagger)')
print(f'  out_of_scope={cls_result["out_of_scope"]}')
```

- [ ] **Step 2: Verify the notebook JSON is valid**

```bash
python -c "import json; json.load(open('eval/06_e2e_pipeline.ipynb')); print('valid JSON')"
```
Expected: `valid JSON`

- [ ] **Step 3: Commit**

```bash
git add eval/06_e2e_pipeline.ipynb
git commit -m "feat: pass geo_scope from /geotag into /classify in E2E notebook"
```

---

## Task 11: Final integration check

- [ ] **Step 1: Run the full test suite**

```bash
python -m pytest tests/ -v --tb=short
```
Expected: all tests PASS.

- [ ] **Step 2: Verify no stale imports remain**

```bash
grep -r "from sklearn\|import sklearn\|TfidfVectorizer" --include="*.py" . | grep -v ".nlp_venv"
grep -r "from nlp.geotagger.disambiguator\|from nlp.summarizer.extractive" --include="*.py" . | grep -v ".nlp_venv"
```
Expected: no output for either command.

- [ ] **Step 3: Smoke-test the import chain**

```bash
python -c "
from nlp.extractor.service import extract_and_embed
from nlp.nli import classify
from nlp.geotagger.service import run as geo_run
from nlp.classifier.service import run as cls_run
from api.main import app
print('all imports ok')
"
```
Expected: `all imports ok`

- [ ] **Step 4: Push**

```bash
git push origin main
```

---

## Self-Review Checklist

**Spec §4 (TextRank extractor):** Covered in Task 2 — `_textrank_extract`, `extract_top_sentences`, tests for max_words and short texts. ✓

**Spec §5 (NLI geo-classification):** Covered in Task 5 — `_classify_geo`, `_pick_city`, `_pick_region`, all three hypothesis constants, fallback logic. ✓

**Spec §6 (Flair NER):** Covered in Task 4 — `ner.py` rewrite, Flair device env var, street regex unchanged. ✓

**Spec §7 (Shared NLI module):** Covered in Task 3 — `nlp/nli.py` with `hypothesis_template` param, `classifier/model.py` shim. ✓

**Spec §8 (Street rescue):** Covered in Task 5 — `rescued` loop in `service.py`. ✓

**Spec §9.1 (geo_scope passthrough):** Covered in Task 6 — `ClassifyRequest.geo_scope`, `run()` accepts it, scope NLI skipped when provided. ✓

**Spec §10 (Dependencies):** Covered in Task 8 — `flair>=0.13` added, `scikit-learn` removed. ✓

**Spec §11 (E2E notebook):** Covered in Task 10 — `geo_scope` added to classify call. ✓

**Spec §13 (all files):** All 12 file changes accounted for across tasks. ✓

**Type consistency check:** `CityHit` defined in `service.py` (Task 5) and used only within `service.py` — no cross-task type mismatch. `Span` dataclass defined in `ner.py` (Task 4) and imported in `service.py` tests (Task 5) as `from nlp.geotagger.ner import Span` — consistent. `nli.classify()` signature defined in Task 3 and called in Task 5 with matching `hypothesis_template` kwarg. ✓
