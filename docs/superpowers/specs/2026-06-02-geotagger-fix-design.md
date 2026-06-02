# Geotagger Fix — Design Spec
**Date:** 2026-06-02  
**Status:** Draft — awaiting user approval  
**Scope:** NER model swap (Flair), shared NLI module, 2-level geo-disambiguation, street routing fix

---

## 1. Complete Current System Map

Understanding what exists before touching anything.

### 1.1 Models loaded at startup

| Module | Model | Framework | Size | Purpose |
|---|---|---|---|---|
| `nlp/encoder.py` | `paraphrase-multilingual-MiniLM-L12-v2` | SentenceTransformers | ~90MB | Embeddings for `/extract` + `/summarize` + dedup |
| `nlp/geotagger/ner.py` | `mrm8488/bert-spanish-cased-finetuned-ner` | HF pipeline | ~110MB | Spanish NER → LOC spans ← **to replace** |
| `nlp/classifier/model.py` | `Recognai/bert-base-spanish-wwm-cased-xnli` | HF zero-shot-classification | ~167MB | Topic NLI, relevance gate, scope NLI ← **to share with geotagger** |
| Ollama sidecar | `gemma4:31b` | Ollama | ~20GB | Headline rewrite + summary generation |

### 1.2 Full pipeline data flow

```
raw article { article_id, text, headline, source }
│
├─► POST /extract
│     TextRank sentence ranking → extract (≤200 words)
│     MiniLM encode(extract) → embedding_raw (384-dim)
│     │
│     ├─► POST /dedup-check-embed          (uses embedding_raw from /extract)
│     │     FAISS cosine search → duplicate_of | indexed
│     │
│     └─► POST /summarize                  (uses extract from /extract)
│           extract → Ollama prompt → { headline, summary }
│           MiniLM encode(headline+summary) → embedding_summary
│           │
│           └─► POST /classify             (uses summary from /summarize)
│                 XNLI relevance gate → out_of_scope?
│                 XNLI multi-label topics → [topic1, topic2, ...]
│                 XNLI scope NLI → geo_scope (national|regional|city)
│
└─► POST /geotag                           (independent of extract/summarize)
      Flair NER → LOC spans
      Gazetteer → P-class (cities) + A-class (regions) + unmatched
      NLI Stage 1 → scope (national|regional|local)
      NLI Stage 2 → winning city OR winning region
      Street regex → geo_streets
      → { geo_scope, geo_cities, geo_streets, geo_points, all_places }
```

### 1.3 API contract changes

One model change: `ClassifyRequest` gains an optional `geo_scope` field (see §9.1).
All other request/response models are unchanged. The only other observable difference
is better geo output quality.

---

## 2. Problems Being Fixed

### Bug 1 — NER model regression (root cause of 5/10 eval failures)
`mrm8488/bert-spanish-cased-finetuned-ner` (WordPiece BERT):
- **1a** Merges city+street into a single span ("Gran Vía de Madrid" instead of separate "Madrid" + "Gran Vía")
- **1b** Produces sub-word artifacts: "Eixample" → `E` + `##ixample`, "Coslada" → `Cos` + `##lada`  
- **1c** "Ayuntamiento de Madrid" → tagged as ORG, "Madrid" never extracted as LOC

**Fix:** Replace with `flair/ner-spanish-large` (F1 90.54, SentencePiece tokenization, document-level context).

### Bug 2 — Street spans from NER never reach `geo_streets`
NER-detected LOC spans with street prefixes (e.g. "Calle Alcalá") are skipped by the
street regex (already in `covered` set), then fail the gazetteer lookup, and become
`[other]` in `all_places` — never entering `geo_streets`.

**Fix:** In `service.py`, rescue no-hit LOC spans that match `_STREET_PREFIX_RE` and
route them into `street_spans` before Stage B2.

### Bug 3 — City disambiguation favours population over text prominence
Madrid (pop ~3.2M) beats Barcelona (pop ~1.6M) even when Barcelona is the article
subject, because the scoring formula over-weights population.

**Fix:** Replace the rule-based `score_candidates` entirely with NLI Stage 2 (see §4).

### Bug 4 — Regional scope not detected
"Comunidad de Madrid" → NER extracts "Madrid" (P-class city), not the A-class admin
region. The scope resolver never sees an A-class entry, defaults to `city`.

**Fix:** Solved by NLI Stage 1 which classifies scope semantically without relying on
GeoNames feature codes (see §4).

---

## 3. Architecture Changes

### Files changed

```
nlp/
  nli.py                        ← NEW: shared NLI wrapper (extracted from classifier/model.py)
  extractor/
    service.py                  ← PATCH: replace TF-IDF sentence ranking with TextRank
    extractive.py               ← MOVE from nlp/summarizer/extractive.py (same code, new home)
  geotagger/
    ner.py                      ← REWRITE: HF pipeline → Flair SequenceTagger
    service.py                  ← REWRITE: absorbs all geo-classification logic;
                                   calls nli.classify() for scope + city selection;
                                   Bug 2 street routing rescue included
    disambiguator.py            ← DELETE: replaced by NLI calls in service.py
  summarizer/
    extractive.py               ← DELETE: moved to nlp/extractor/extractive.py
  classifier/
    model.py                    ← SIMPLIFY: thin import shim over nlp.nli

api/models.py                   ← PATCH: add optional geo_scope field to ClassifyRequest
requirements.txt                ← add flair>=0.13; remove sklearn (TF-IDF gone); keep sumy
Dockerfile                      ← update model pre-pull
api/main.py                     ← add flair device init to lifespan
```

### Files NOT changed

`gazetteer.py`, all routers except classify (no router change — model change only),
`encoder.py`, `dedup/`, `config/topics.yaml`, `api/warmth.py`.

---

## 4. Extractor: TF-IDF → TextRank (nlp/extractor/service.py)

### Why

TF-IDF in `extractor/service.py` fits a vectoriser on 5–10 sentences from the same
article. With such a small corpus the IDF component is meaningless — it degenerates
into a raw word-frequency ranker. `nlp/summarizer/extractive.py` already has a correct
TextRank implementation (sumy `TextRankSummarizer`) that considers inter-sentence
similarity via a graph, producing more coherent and representative sentence selections.

### Change

Move `nlp/summarizer/extractive.py` → `nlp/extractor/extractive.py` (no code change,
just relocation to the module that actually uses it).

Replace the TF-IDF body in `extractor/service.py` with a call to `extract_top_sentences`,
preserving the same `max_words` contract:

```python
# after
from nlp.extractor.extractive import extract_top_sentences

def _textrank_extract(text: str, max_words: int) -> str:
    sentences = _split_sentences(text)
    if len(sentences) <= 2:
        return text
    avg_words = sum(len(s.split()) for s in sentences) / len(sentences)
    n = max(1, int(max_words / max(avg_words, 1)))
    return extract_top_sentences(text, n)
```

### What stays the same

- `extract_and_embed()` signature is unchanged
- `/extract` response fields `extract` and `embedding_raw` are unchanged
- Ollama receives the same `extract` string (better sentences, same field)
- Dedup embedding computed from TextRank extract — existing index remains valid,
  thresholds unchanged

### Dead code removed

`sklearn` import and `_tfidf_extract` deleted from `extractor/service.py`.
`numpy` import in extractor also removed (was only used by TF-IDF).

---

## 5. Geo-Classification in service.py (replaces disambiguator.py)

`disambiguator.py` is deleted. The NLI scope + city classification moves directly into
`service.py`, which already holds all the context needed (spans, gazetteer hits, text,
headline). This removes an unnecessary indirection layer — the logic is two `nli.classify()`
calls, not a separate module's responsibility.

All NLI calls go through the shared `nlp.nli.classify()`.

### 5.1 Inputs (inside service.py `run()`)

After Stage B1 gazetteer lookups, `service.py` has:
- `spans_with_geo`: list of `(span_text, [GeoEntry, ...])` — same as today
- `premise = f"{headline}. {full_text}"` — truncation handled by HF pipeline

Candidate pools derived from `spans_with_geo`:
- `city_pool`: spans where best GeoEntry has `feature_class == "P"` AND matches a cities-DB entry
- `region_pool`: spans where best GeoEntry has `feature_class == "A"`

### 5.2 Stage 1 — Geographic scope (always runs)

**Goal:** Is this article about a single city, a region, or a national topic?

**Method:** `nli.classify(premise, labels=[h_local, h_regional, h_national], multi_label=False)`

`multi_label=False` runs a softmax over the three hypotheses — the model picks the most
entailed one. This is correct here because the three scopes are mutually exclusive.

**Hypotheses** (in Spanish, matching the article language and the XNLI model's training):

```python
H_LOCAL = (
    "Este artículo describe actuaciones, obras o iniciativas "
    "de un ayuntamiento o municipio concreto de España."
)

H_REGIONAL = (
    "Este artículo describe políticas o actuaciones de una comunidad autónoma, "
    "diputación provincial o región de España que afectan a varios municipios."
)

H_NATIONAL = (
    "Este artículo describe una política, ley, normativa o acontecimiento "
    "de alcance estatal en España, sin limitarse a una ciudad o región concreta."
)
```

**Why these work:**

| Test case | Key text signal | Expected winner |
|---|---|---|
| geo-001: "El Ayuntamiento de Madrid ha inaugurado…" | "Ayuntamiento" = municipal actor | H_LOCAL |
| geo-003: "Barcelona supera a Madrid en km de carril bici" | city comparison, no regional body | H_LOCAL |
| geo-009: "La Comunidad de Madrid ha aprobado un plan…" | "Comunidad de Madrid" = autonomous community | H_REGIONAL |
| geo-005: "El Ministerio de Transportes ha presentado…" | "Ministerio" = central government | H_NATIONAL |

**Output → `geo_scope`:**
- `"local"` → proceed to Stage 2a
- `"regional"` → proceed to Stage 2b
- `"national"` → return `("national", None)`, skip Stage 2

**Fallback:** If `max(scores) < SCOPE_CONFIDENCE_THRESHOLD` (default 0.35), fall back to
heuristic: `city` if city_pool non-empty, `regional` if region_pool non-empty, else `national`.

### 5.3 Stage 2a — City selection (only if Stage 1 = local)

**Skip condition:** If `len(city_pool) <= 1`, return the single candidate directly (or None).

**Goal:** Which city is this article primarily about?

**Method:** `nli.classify(premise, labels=city_names, multi_label=False, hypothesis_template=CITY_TEMPLATE)`

The `hypothesis_template` is passed to the HF pipeline so the model tests:
`"Este artículo describe principalmente hechos o iniciativas en {}."` for each city name.

```python
CITY_TEMPLATE = "Este artículo describe principalmente hechos o iniciativas en {}."
```

**Why this template works:**

"En Barcelona" is entailed by "Barcelona ha aprobado…" or "el ayuntamiento de Barcelona…".
It's NOT entailed by "Barcelona cuenta con más km que Madrid" because that sentence is
*about the comparison*, not about Barcelona's actions. That said, the model will score
Barcelona higher because Barcelona is the grammatical subject of most sentences.

**Candidates:** deduplicated city names from `city_pool`, mapped through `_match_city` to
known cities in the DB. Maximum 10 candidates; if more exist, pre-filter to the 10 with
highest `entry.population` before running NLI (avoids tiny obscure settlements dominating
the label set).

**Output:** `CityHit` for the winning city, or `None` if all scores below threshold.

### 5.4 Stage 2b — Region selection (only if Stage 1 = regional)

**Skip condition:** If `len(region_pool) <= 1`, return the single candidate directly.

**Goal:** Which autonomous community / province is this article about?

**Method:** same as 2a but with a different template:

```python
REGION_TEMPLATE = (
    "Este artículo trata principalmente sobre "
    "la comunidad autónoma, provincia o región de {}."
)
```

**Output:** `geo_region` string (name of the region), no `CityHit`.

### 5.5 Combined result (inside service.py)

After both stages, `service.py` holds:
- `geo_scope` ∈ `{"national", "regional", "city"}` — Stage 1's `"local"` maps to `"city"`
  to match the existing API contract
- `winning_city` — a `GeoEntry` + cities-DB dict pair, or `None`
- `geo_region` — an A-class entry name, or `None`

`CityHit` dataclass moves into `service.py` (or is inlined) since it's no longer shared
with a separate disambiguator module. The `_resolve_scope` function in `service.py` is
**deleted** — scope now comes directly from Stage 1 NLI output.

---

## 6. NER Module Rewrite (ner.py)

### 6.1 Flair API

```python
import os
import torch
import flair
from flair.models import SequenceTagger
from flair.data import Sentence

# Device — env var NER_DEVICE, default "cpu"
# Set before any Flair model is loaded; Flair reads flair.device at load time.
flair.device = torch.device(os.environ.get("NER_DEVICE", "cpu"))

_tagger: SequenceTagger | None = None

def _ensure_loaded() -> None:
    global _tagger
    if _tagger is None:
        _tagger = SequenceTagger.load("flair/ner-spanish-large")
```

### 6.2 Entity extraction

```python
def _flair_spans(text: str) -> list[Span]:
    sentence = Sentence(text, use_tokenizer=True)
    _tagger.predict(sentence)
    return [
        Span(
            text=entity.text,
            label=entity.tag,           # "LOC", "PER", "ORG", "MISC"
            start_char=entity.start_position,
            end_char=entity.end_position,
        )
        for entity in sentence.get_spans("ner")
        if entity.tag in _KEEP_LABELS   # {"LOC"}
    ]
```

No sub-word handling needed: Flair uses SentencePiece internally and always returns
clean, complete words.

### 6.3 Street regex stage (unchanged)

The `_STREET_RE` regex post-pass runs identically after Flair. The `covered` set prevents
double-counting when Flair already tagged a street span as LOC.

### 6.4 Startup in main.py

Add to lifespan:
```python
import flair as _flair_lib
import torch as _torch
_flair_lib.device = _torch.device(os.environ.get("NER_DEVICE", "cpu"))
_ner._ensure_loaded()   # already called — no line change needed
```

---

## 7. Shared NLI Module (nlp/nli.py)

Currently `nlp/classifier/model.py` is the only NLI entry point. `geotagger/service.py`
now also needs NLI calls. Rather than importing across modules (geotagger importing
from classifier), extract to a shared module.

```
nlp/
  nli.py          ← NEW: pipeline singleton + classify()
  classifier/
    model.py      ← becomes a 3-line import shim
  geotagger/
    service.py    ← imports from nlp.nli directly
```

`nlp/nli.py` interface:

```python
def classify(
    text: str,
    labels: list[str],
    multi_label: bool,
    hypothesis_template: str = "{}",   # "{}" = labels are already full hypotheses
) -> dict:
    """Wraps HF zero-shot-classification pipeline.
    Returns {'labels': [...], 'scores': [...]} sorted descending by score.
    """
```

`nlp/classifier/model.py` becomes:
```python
from nlp.nli import classify   # re-export for backward compat
```

`main.py` lifespan warmup: `_cls_model._ensure_loaded()` → replace with `nli._ensure_loaded()`.
The `_ensure_loaded` function lives in `nlp/nli.py`.

---

## 8. Street Routing Fix (service.py — Bug 2)

After the gazetteer lookup loop, before Stage B2:

```python
# Rescue LOC spans that NER caught but gazetteer doesn't know —
# if they carry a street prefix, treat them as street spans.
rescued: list[ner.Span] = []
for s in loc_spans:
    if not gazetteer.lookup(s.text) and gazetteer._STREET_PREFIX_RE.match(s.text):
        rescued.append(s)

for r in rescued:
    loc_spans.remove(r)
    # Wrap in a street-hint Span for the B2 resolver
    street_spans.append(ner.Span(
        text=r.text, label=r.label,
        start_char=r.start_char, end_char=r.end_char,
        hint="street",
    ))
```

This rescues "Calle Alcalá", "Plaza de Cibeles", "Avenida Diagonal" etc. that NER
detected but that have no geonames entry.

---

## 9. Cross-Cutting Concerns

### 9.1 Dual geo_scope — two sources of truth

**Current state:** Both `/geotag` and `/classify` independently compute `geo_scope`.

- `/geotag` computes scope from geographic entity evidence (NER + NLI)
- `/classify` computes scope from the article's *policy level* (using `topics.yaml` `scope_hypotheses`)

**These are different things:**
- Geotagger scope = "where does this physically happen?" (geographic authority)
- Classifier scope = "at what administrative level is this policy?" (editorial signal)

They should generally agree but can legitimately diverge. Example: a Barcelona cycling
study covered by a national newspaper → geotagger says `city`, classifier might say
`national` (policy significance). Both are valid in their dimension.

**Resolution:** `ClassifyRequest` gains an optional `geo_scope: str | None = None` field.
When the E2E pipeline has already called `/geotag`, it passes `geo_scope` from that
response into `/classify`. The classifier skips its own `_scope_pass` NLI when
`geo_scope` is provided, using it directly. When `geo_scope` is absent (e.g. quick
classification without geotag), the classifier runs its own scope NLI as today.

This makes the geotagger the authoritative geographic source while preserving the
classifier's independent scope signal for callers that don't run geotag.

### 9.2 XNLI model token limit

`Recognai/bert-base-spanish-wwm-cased-xnli` is a BERT-base model: 512 tokens max.
NLI input = `[CLS] premise [SEP] hypothesis [SEP]`. A typical hypothesis is ~25 tokens,
leaving ~480 tokens for the premise. Most news articles fit; long ones are silently
truncated by HF pipeline when `truncation=True` (already the default).

No code change needed. Document the behaviour.

### 9.3 GPU / device configuration

| Model | Current device | With this spec |
|---|---|---|
| MiniLM encoder | CPU | CPU (unchanged) |
| XNLI (NLI) | CPU | CPU (unchanged) |
| Flair NER | N/A | CPU default, `NER_DEVICE=cuda:0` for GPU |
| Ollama | GPU (both L40s) | GPU (unchanged) |

Given GPU 0 has ~2.2 GB free and Flair's XLM-R-large weights are ~1.4 GB FP32, GPU
offload is feasible but tight. Default to CPU; operators can set `NER_DEVICE=cuda:0`.

### 9.4 Warmup (main.py lifespan)

No structural change to the warmup sequence. Flair `_ensure_loaded()` replaces the HF
pipeline load. `mark_warm("geotag")` fires after both `_ner._ensure_loaded()` and
`_geo_svc.load()` complete.

---

## 10. Dependencies and Dockerfile

### requirements.txt

Add:
```
flair>=0.13
```

Note: Flair pulls in `torch` (already present via transformers), `gensim`, and several
other packages. Total image size increase ~200–400 MB.

Keep (now used by TextRank extractor):
```
sumy
```

Remove (TF-IDF path deleted):
```
scikit-learn
```

### Dockerfile model pre-pull

Replace:
```dockerfile
# NER: Spanish BERT NER
RUN python -c "from transformers import pipeline; pipeline('token-classification', model='mrm8488/bert-spanish-cased-finetuned-ner', aggregation_strategy='simple')"
```

With:
```dockerfile
# NER: Flair Spanish large (XLM-R + character LM)
RUN python -c "from flair.models import SequenceTagger; SequenceTagger.load('flair/ner-spanish-large')"
```

---

## 11. Eval Notebook Impact

The only API model change is an optional `geo_scope` field on `ClassifyRequest` —
backward compatible, no notebook changes required. The E2E notebook (`06_e2e_pipeline.ipynb`)
should be updated to pass `geo_scope` from the geotag response into the classify call.
The geotag eval notebook (`03_geotag_eval.ipynb`) output format is unchanged; accuracy
should improve for all 6 currently-failing cases.

Expected pass rate: 8–10/10 (geo-006 "La Rambla → Barcelona" remains uncertain — the
NLI model may not know La Rambla is a Barcelona landmark without gazetteer enrichment).

---

## 12. Open Question

**geo-006 "La Rambla":** Flair will return "Rambla" as a LOC span. The gazetteer has
no geonames entry for "La Rambla" as a Barcelona street (it's a proper-noun street name
not in geonames). Stage 1 NLI will likely say `local` (one specific place, one city),
but Stage 2 city pool will be empty (no P-class candidates). The result would be
`geo_scope="city", geo_cities=[]` — correct scope, no city assigned.

**Resolution options:**
1. Add "La Rambla" to the street index manually (config change, not code)
2. Accept as a known limitation (still an improvement over current `national` scope)
3. Future: a Barcelona-specific landmark lookup table in the gazetteer

This is out of scope for this spec. Option 2 is recommended for now.

---

## 13. Summary of Changes per File

| File | Change | Reason |
|---|---|---|
| `nlp/nli.py` | **Create** | Shared NLI singleton; avoids cross-module coupling |
| `nlp/extractor/extractive.py` | **Create** (move) | TextRank implementation relocated from summarizer/ |
| `nlp/extractor/service.py` | **Patch** | Replace TF-IDF with TextRank; better sentence selection |
| `nlp/geotagger/ner.py` | **Rewrite** | Flair replaces HF pipeline; fixes Bugs 1a/1b/1c |
| `nlp/geotagger/service.py` | **Rewrite** | Absorbs NLI scope+city classification; street rescue; fixes Bugs 2/3/4 |
| `nlp/geotagger/disambiguator.py` | **Delete** | Logic moved into service.py |
| `nlp/summarizer/extractive.py` | **Delete** (moved) | Now lives at nlp/extractor/extractive.py |
| `nlp/classifier/model.py` | **Simplify** | Thin import shim over `nlp.nli` |
| `api/models.py` | **Patch** | Add optional `geo_scope` to `ClassifyRequest` |
| `api/main.py` | **Patch** (~3 lines) | Flair device init + nli warmup |
| `requirements.txt` | **Update** | Add flair; remove scikit-learn; keep sumy |
| `Dockerfile` | **Update** | Swap model pre-pull |
