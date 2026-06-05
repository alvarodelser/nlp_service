# NLI — Implementation Doc

**Module:** `nlp/nli.py` · **Status:** Reworked (absorbs + retires `nlp/classifier/`)
**Authoritative source:** `docs/superpowers/specs/2026-06-05-nlp-two-pipelines-design.md` §3.2

## Purpose

`nli` is the project's zero-shot entailment **primitive**. It scores hypotheses against a text
and **returns scores only** — it makes no in/out-of-scope decision, holds no taxonomy, and
knows nothing about topics, scope, or use cases. Every consumer (relevance gating, topic
assignment, scope assignment, the geotagger's typing/tie-break) builds its own verdict on top of
the scores.

The old `classifier` module bundled the primitive **and** the news-domain orchestration
(relevance gate, blacklist, topic filtering, nested scope) **and** a YAML taxonomy. This rework
keeps only the primitive here and **moves all orchestration + taxonomy out to the news pipeline
orchestrator** (doc 07).

Two functions:

- **`classify(text, labels, multi_label=True, hypothesis_template="{}")`** — the low-level
  pipeline wrapper (unchanged). `multi_label=True` scores each label independently;
  `multi_label=False` does the mutually-exclusive softmax used for *pick-the-best-of-N* (the
  geotagger's city/region tie-break and the orchestrator's scope selection).
- **`score(text, hypotheses, threshold=None, blacklist=False, hypothesis_template="{}")`** —
  the **new contract**: ordered, independently-scored hypotheses with optional short-circuit.

---

## The `score()` contract

Runs hypotheses **in the given order** and returns a list of `{hypothesis, score}` for the ones
it actually ran (input order preserved). The caller interprets the verdict.

| `threshold` | `blacklist` | Behaviour | Caller verdict |
|---|---|---|---|
| `None` | — | Run **all** hypotheses, return every score | OR (any pass) or inspect |
| set | `False` | Stop at the **first score below** threshold (the failing one is included) | **AND** — passed iff no returned score is below threshold |
| set | `True` | Stop at the **first score at/above** threshold (the tripping one is included) | **AND-exclusion** — clear iff no returned score reaches threshold |

The returned list always includes the hypothesis that triggered the short-circuit, so the caller
can see *which* one broke. Hypothesis order is the caller's efficiency lever: put the
most-likely-to-fail first for a whitelist AND, the most-likely-to-trip first for a blacklist.

### How the news orchestrator (doc 07) uses it

```python
# Whitelist gate — ANY in-scope topic must pass (OR): no threshold, then check outside.
wl = nli.score(headline_summary, IN_SCOPE_HYPOTHESES)            # all scores
if not any(s["score"] >= REL_THRESHOLD for s in wl):
    return out_of_scope

# Blacklist gate — ALL must stay clear (AND-exclusion): threshold + blacklist=True.
bl = nli.score(headline_summary, BLACKLIST_HYPOTHESES, threshold=BL_THRESHOLD, blacklist=True)
if any(s["score"] >= BL_THRESHOLD for s in bl):                  # something tripped
    return out_of_scope
```

(Topic assignment uses `score()` with no threshold and keeps labels above the orchestrator's
topic threshold; scope selection uses `classify(..., multi_label=False)` and takes the argmax —
see doc 07.)

---

## What changes and what does not

| Component | Status |
|---|---|
| `nlp/nli.py` — `classify()` | **Unchanged** (geotagger + scope ranking depend on it) |
| `nlp/nli.py` — `score()` | **New** function appended |
| `nlp/nli.py` — `_MODEL_NAME` | Made env-configurable (`NLI_MODEL`), default unchanged |
| `nlp/classifier/model.py` | **Deleted** (was just `from nlp.nli import _ensure_loaded, classify`) |
| `nlp/classifier/service.py` | **Deleted** — relevance/blacklist/topic/scope logic moves to doc 07 |
| `nlp/classifier/taxonomy.py` | **Deleted** — `topics.yaml` becomes orchestrator config (doc 07) |
| `nlp/classifier/__init__.py` + dir | **Deleted** (whole module retired) |
| `api/routers/classify.py` | **Replaced** by `api/routers/nli.py` (`POST /nli`) |
| `api/models.py` — `Classify*`, `SourceProfile` | **Deleted/moved** (orchestrator config, doc 07) |
| `api/models.py` — `NliRequest/Response`, `ScorePair` | **New** |
| `api/main.py` | Swap `classify_router`→`nli_router`; drop `classifier` warmup import |
| `config/topics.yaml` | **Stays a file**, but is loaded by the orchestrator now, not this module |

> Confirm the blast radius before deleting `nlp/classifier/`:
> `grep -rn "nlp.classifier\|from nlp.classifier" nlp api tests` should only show the files
> listed above (`model.py`, `service.py`, `api/routers/classify.py`, `api/main.py`, and the
> classifier tests). The geotagger imports `from nlp import nli` directly — it does **not** go
> through `classifier`, so it is unaffected.

---

## File layout (after rework)

```
nlp/
  nli.py              ← classify() unchanged + score() appended; NLI_MODEL env
api/
  routers/
    nli.py            ← POST /nli  (was classify.py)
config/
  topics.yaml         ← unchanged file; now read by the orchestrator (doc 07), not here
```

`nlp/classifier/` is removed entirely.

---

## `nlp/nli.py` — appended `score()`

The existing module already loads the pipeline and exposes `classify()`. Append `score()` and
make the model name configurable. Nothing above is modified except the one constant.

```python
import os
from transformers import pipeline

_MODEL_NAME = os.environ.get("NLI_MODEL", "Recognai/bert-base-spanish-wwm-cased-xnli")
_pipeline = None


def _ensure_loaded() -> None:
    global _pipeline
    if _pipeline is None:
        _pipeline = pipeline("zero-shot-classification", model=_MODEL_NAME)


def classify(text, labels, multi_label=True, hypothesis_template="{}") -> dict:
    """Unchanged. Raw pipeline dict {'labels','scores','sequence'}, sorted desc by score."""
    _ensure_loaded()
    assert _pipeline is not None
    return _pipeline(text, candidate_labels=labels,
                     multi_label=multi_label, hypothesis_template=hypothesis_template)


def score(
    text:                str,
    hypotheses:          list[str],
    threshold:           float | None = None,
    blacklist:           bool = False,
    hypothesis_template: str = "{}",
) -> list[dict]:
    """Ordered, independently-scored hypotheses with optional short-circuit.

    Returns [{"hypothesis": str, "score": float}, ...] in input order for the hypotheses
    actually run. See the contract table in 02-nli.md.
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
```

> **Why per-hypothesis calls when a threshold is set:** the zero-shot pipeline with
> `multi_label=True` scores each hypothesis independently, so running them one at a time yields
> the same scores as a batch while enabling early exit. With no threshold there is nothing to
> short-circuit, so the batched single call is used.

---

## Why `classify()` stays (the mutually-exclusive mode)

`score()` covers independent scoring + thresholds. Two consumers need *pick-the-best-of-N*
instead, which is `classify(..., multi_label=False)` (softmax across labels, then argmax):

- **Geotagger** (doc 03): choosing one city among candidates, one region among candidates, and
  typing a toponym `region`/`city`/`street·loc`.
- **News orchestrator** (doc 07): scope pass 1 (`city`/`regional`/`national`) and pass 2 (which
  detected region/city).

These call `nli.classify()` **in-process**. The HTTP surface exposes only `/nli` (the `score()`
contract); if an out-of-process ranking endpoint is ever needed, add `/nli/rank` then — not now
(YAGNI).

---

## API (`api/routers/nli.py`)

Replaces `classify.py`. Stateless, no taxonomy load.

```python
import logging
from fastapi import APIRouter, HTTPException

from api.models import NliRequest, NliResponse
from api.warmth import mark_warm
from nlp import nli

log = logging.getLogger(__name__)
router = APIRouter()


@router.post("/nli", response_model=NliResponse)
def run_nli(req: NliRequest) -> NliResponse:
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must be non-empty")
    if not req.hypotheses:
        raise HTTPException(status_code=422, detail="hypotheses must be non-empty")
    scores = nli.score(
        text=req.text,
        hypotheses=req.hypotheses,
        threshold=req.threshold,
        blacklist=req.blacklist,
        hypothesis_template=req.hypothesis_template,
    )
    mark_warm("nli")
    return NliResponse(request_id=req.request_id, scores=scores)
```

### Models (`api/models.py`)

Delete `ClassifyRequest`, `ClassifyResponse`, `SourceProfile` (the last moves to orchestrator
config, doc 07). Add:

```python
class NliRequest(BaseModel):
    request_id:          str | None = None
    text:                str
    hypotheses:          list[str]
    threshold:           float | None = None
    blacklist:           bool = False
    hypothesis_template: str = "{}"

class ScorePair(BaseModel):
    hypothesis: str
    score:      float = Field(ge=0.0, le=1.0)

class NliResponse(BaseModel):
    request_id: str | None = None
    scores:     list[ScorePair]      # input order; length < len(hypotheses) ⇒ short-circuited
```

### `api/main.py`

- Router registration: replace
  `from api.routers import classify as classify_router` / `app.include_router(classify_router.router)`
  with `from api.routers import nli as nli_router` / `app.include_router(nli_router.router)`.
- Remove `extract_router` (retired in doc 01) at the same time.
- Warmup: drop `from nlp.classifier import service as _cls_svc` and any `_cls_svc.load()`
  (no taxonomy to preload). Keep `_nli._ensure_loaded()` — the comment
  "shared by classifier + geotagger" becomes "shared by /nli + geotagger".

---

## What moves to the orchestrator (doc 07) — nothing lost

The retired `classifier/service.py` + `taxonomy.py` logic is **not deleted, it relocates**. The
news orchestrator (doc 07) will own:

- Loading `config/topics.yaml` (the `Taxonomy` dataclass: `labels`, `blacklist_labels`,
  `score_threshold`, `top_k`, `relevance_hypothesis`, `relevance_threshold`,
  `blacklist_threshold`, `scope_hypotheses`, `scope_threshold`).
- The **relevance gate** → `nli.score(summary, [relevance_hypothesis])` then compare to
  `relevance_threshold`.
- The **topic pass** → `nli.score(tag_prefixed_summary, labels)` then keep labels ≥
  `score_threshold`, top-`top_k`. (The "Artículo buscado por: …" search-tag prefix logic moves
  with it.)
- The **blacklist gate** → `nli.score(summary, blacklist_labels, threshold=blacklist_threshold,
  blacklist=True)`.
- The **nested scope** → `nli.classify(premise, [scope hyps], multi_label=False)` for
  city/regional/national, then a second `classify()` over detected regions/cities, with the
  `geo_cities`/`source_profile` context assembled by the orchestrator.

Doc 07 specifies these in full; this section exists so the migration is traceable.

---

## Configuration

| Env var | Default | Description |
|---|---|---|
| `NLI_MODEL` | `Recognai/bert-base-spanish-wwm-cased-xnli` | Zero-shot XNLI model |

`TOPICS_YAML_PATH` is no longer read by this module — it moves to the orchestrator's config.

---

## Dependencies

- `transformers` — already in requirements (zero-shot pipeline).
- No new packages. The XNLI model is loaded once (`_ensure_loaded`) and shared by `/nli` and the
  geotagger.
