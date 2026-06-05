# NLP Service — Two-Pipeline Design

**Status:** Authoritative design source. Date: 2026-06-05.

This document is the single authoritative description of the refactored NLP service. The
per-module implementation documents (rewritten over `docs/entity-pipeline/*`) and the two
orchestrator specs are derived from it. If they disagree, this document wins.

---

## 1. Guiding principles

- **Agnostic, stateless modules.** Each module does one job behind a typed contract. It knows
  nothing about documents, use-cases, IDs, databases, or pipeline order. No module hardcodes
  domain-specific attribute rules; schema/attributes are opaque pass-throughs (see
  `[[project_schema_variable]]`, `[[project_normalizer_edge_semantics]]`).
- **Orchestrators own all state.** Two synchronous Python orchestrators — one per use case —
  drive the modules in order and own everything stateful: Weaviate/Neo4j/Postgres I/O, MinHash,
  the `merge` operation, schema/collection creation, and intermediate-result retention until the
  final upsert.
- **One module writes prose.** Only the `summarizer` generates free text. Extraction, typing,
  resolution, and dedup copy verbatim evidence or decide structure.
- **Single vector space.** Everything embeds with **bge-m3 (1024-dim)** via the existing
  `vectorizer` service. The legacy MiniLM-384 article embedding is retired.

---

## 2. Module inventory

| Module | Role | Status |
|---|---|---|
| `summarizer` | LLM text generation with **stored named profiles**; extractive pre-step baked in | Reworked |
| `nli` | Pure hypothesis scoring | Renamed from `classifier` |
| `geotagger` | Pure toponym detection → typing → resolution | Reworked |
| `ner` | LLM entity + relation + attribute extraction | New (was `entity_extractor`) |
| `resolver` | Intra-document merge-on-name of entities + relations | Keep name |
| `dedup` | Cross-document dedup — two endpoints (item, corpus) | Reworked |
| `vectorizer` | bge-m3 embedding service | **Existing, out of scope** — reference only |

Retired: the old TextRank `extractor` module (absorbed into `summarizer`); the
`classifier` taxonomy/scope/threshold orchestration logic (moves into the orchestrators); the
FAISS + local-persistence article dedup path (articles move fully to Weaviate); MiniLM-384.

---

## 3. Module contracts

### 3.1 `summarizer`

LLM generation driven by **stored named profiles**. A profile bundles a prompt template, an
output JSON schema, and validation/length rules, all versioned inside the module.

- `summarize(profile: str, fields: dict) -> dict`
- Profiles:
  - `article` — in: `title`, `text` → out: `headline`, `summary`.
  - `entity_desc` — in: `name`, `type`, `subtype?`, `evidence[]` → out: `description` (Spanish).
  - `relation_desc` — in: endpoints, `type`, `evidence[]`, `attributes` → out: `description`.
  - `aggregate` — in: a cluster's accumulated descriptions/evidence → out: cross-document
    `description`.
- **Extractive pre-step is baked in.** When `fields` text exceeds the model token budget, the
  module runs the TextRank extractive reducer (migrated from the old `extractor`) before the LLM
  call. Below the budget, the raw text is used. The caller never invokes a separate extract step.
- Output is constrained-decoded against the profile's schema; a single tightened retry on
  validator failure (existing `article` behaviour preserved).

### 3.2 `nli`

Pure NLI scoring. **Returns scores only**; the orchestrator interprets verdicts.

- `score(text: str, hypotheses: list[str], threshold: float | None = None, blacklist: bool = False) -> dict`
- Runs hypotheses **in order**, returns `{hypothesis: score}` for the ones actually run.
- Semantics:
  - `threshold = None` → run all hypotheses, return every score. (Caller does OR or inspects.)
  - `threshold` set, `blacklist = False` → **stop at the first score below threshold** and
    return scores so far. (AND / all-must-pass: a failure short-circuits the rest.)
  - `threshold` set, `blacklist = True` → **stop at the first score at/above threshold** and
    return scores so far. (AND-exclusion: a violation short-circuits the rest.)
- Hypothesis order is the caller's lever for short-circuit efficiency (most-likely-to-fail
  first for whitelists; most-likely-to-trip first for blacklists).
- Same primitive serves relevance gating, topic assignment, scope assignment, and the
  geotagger's internal typing/tie-break calls.

### 3.3 `geotagger`

Pure toponym resolution. **No scope decision** (scope moves to the news orchestrator's `nli`
step). Depends on the `nli` module, the **gazetteer** (regions + cities snapshot), and
**Postgres** (street geometry / edge ids).

- `geotag(text, headline?, source?) -> {entities: [...]}`
- Steps:
  1. **Detect** toponym spans (NER + regex).
  2. **Type first** — label each span `region` / `city` / `street·loc`. Streets are pre-typed by
     regex (high precision); ambiguous spans are typed by a quick internal `nli` call.
  3. **Resolve in order**: regions → cities (gazetteer); streets + locations (Postgres /
     gazetteer).
     - **Region / city**: resolve against the gazetteer.
     - **Street → city imputation**: query **Postgres** for candidate cities = cities whose
       street DB contains that street name (geometry exists). Among candidates, assign the street
       to the **geographically closest city detected in the article text** (compare city
       centroids). Exactly one candidate → use it. No detected cities → source prior, else
       `city = null`. No street-DB match → `city = null`. Resolved streets carry `edge_ids`
       (geometry already in Postgres) — **no coordinates returned**.
     - **Location (sub-city POI)**: coordinates from the **gazetteer only**; if found, also
       impute its `city` from the lat/lon. POIs absent from the gazetteer return `city`
       (if imputable) but no coordinates.
- **Output**: a flat list of place entities, each with `text`, `type`
  (`region`/`city`/`street`/`location`), resolved identifiers, `city` for sub-city entities,
  optional `lat`/`lon` for locations, `edge_ids` for streets. Nothing about scope.

### 3.4 `ner`

LLM extraction of entities, relations, and attributes from a single chunk against an inline
schema. (Naming note: chosen as `ner` per owner; collides conceptually with the classic term
and with `geotagger/ner.py` — flagged, not blocking.)

- `extract(text: str, schema: ExtractionSchema) -> {entities: [...], relations: [...]}`
- `ExtractionSchema` (inline, per call): entity types/subtypes/attributes (each with name +
  LLM-facing description; attributes also carry a datatype); relations with name + description +
  subject/object type restrictions.
- Steps: (1) pronoun preprocessing (resolve antecedents to names, on the chunk as received),
  (2) one constrained LLM call emitting a discriminated-union grammar (one branch per type;
  undeclared types/attributes/datatypes unrepresentable), (3) post-validation of datatypes,
  subtypes, attributes, and relation subject/object constraints.
- **Output**: entities with verbatim `evidence` (the sentence they appear in); relations with
  `type`, `attributes`, `confidence`, and verbatim `evidence`. No offsets, no IDs, no
  doc-awareness — the orchestrator owns those.

### 3.5 `resolver`

Stateless, intra-document merge-on-name of a document's entities and relations.

- `resolve(entities: [...], relations: [...]) -> {entities: [...], relations: [...]}`
- Algorithm: pre-merge by normalized name + type (pure Python) → LLM refine on the reduced
  candidate set only (definite descriptions, same-name splits) → rewrite relations to canonical
  names and collapse overlap-duplicates deterministically.
- **Relations are never aggregated** — distinct evidence means distinct instances; attributes
  pass through unchanged (`[[project_normalizer_edge_semantics]]`).
- Output: reduced canonical entities (with `names[]` aliases, collected `evidence[]`, assigned
  id) + relations rewritten to canonical endpoints.

### 3.6 `dedup` — two endpoints

- **`/dedup/item`** (single new item): given an embedding + type, query Weaviate for top-K
  candidates → if best similarity ≥ merge threshold, return `match` + target id; if in the
  middle band, **LLM double-check** (the 2-step procedure validated in
  `eval/05_dedup_eval.ipynb`) → `match` / `no-match`; below the low band, `no-match`. Returns a
  decision, not a write — the orchestrator performs the merge.
- **`/dedup/corpus`** (whole collection): given a Weaviate collection + query restricted to
  entities or relations, cluster the set — direct-merge below the tight threshold, LLM
  double-check in the band above — and **return clusters**. Run separately for entities and for
  relations. The orchestrator turns clusters into merges + aggregate summaries.

### 3.7 `vectorizer` (existing — out of scope)

Already deployed and working. Embeds text to **bge-m3 (1024-dim)**. This design only references
it; it is not (re)built here. Embedding composition must be identical at insert and query time
for any given collection.

---

## 4. Pipeline 1 — News orchestrator (synchronous Python)

```
article (title, text, source, url, date)
  │
  ├─ MinHash vs Weaviate ── duplicate? ──► merge(): append source+url, keep oldest date ──► DONE
  │
  ├─ summarizer[article]              → headline, summary
  │
  ├─ nli (scope gate)
  │     whitelist hypotheses (no threshold) → orchestrator: ANY pass? (OR)   [threshold ⇒ AND]
  │     blacklist hypotheses (threshold, blacklist=true) → orchestrator: ALL clear? (AND-excl)
  │     fail either gate ──► DONE (out of scope)
  │
  ├─ vectorizer(headline + summary)   → embedding (bge-m3)
  │
  ├─ dedup/item(embedding)            → match? ──► merge() ──► DONE
  │
  ├─ geotagger(headline, text, source)→ resolved places
  │
  ├─ resolver(places)                 → de-duplicated place list
  │
  ├─ nli (topics)   topic hypotheses  → orchestrator assigns topics
  │
  ├─ nli (scope)    nested:
  │     pass 1: city / regional / national (detected geotags as extra context)
  │     pass 2: if regional or city → hypotheses over detected regions/cities → highest wins
  │
  └─ upsert Weaviate (article + minhash signature + topics + scope + places)
```

The taxonomy that the old `classifier` baked in (in-scope topics, blacklist, scope hypotheses,
thresholds) becomes **orchestrator configuration**, passed into stateless `nli` calls.

---

## 5. Pipeline 2 — Relation extraction orchestrator (synchronous Python)

```
chunks (text) + ExtractionSchema
  │
  ├─ ner(chunk, schema)  per chunk    → entities (+evidence), relations (+attrs, type, confidence)
  │     (orchestrator flattens chunks; relation endpoints → entity names)
  │
  ├─ resolver(entities, relations)    → reduced entities + relations (ids, canonical names)
  │
  ├─ summarizer[entity_desc | relation_desc]  per node/edge → description
  │
  ├─ vectorizer(description)          → embedding (bge-m3)
  │
  ├─ upsert Weaviate (per-document entities + relations)
  │
  ├─ dedup/corpus(entities)           → entity clusters
  ├─ dedup/corpus(relations)          → relation clusters
  │
  ├─ summarizer[aggregate]  per cluster → cross-document description
  │
  └─ upsert final Weaviate collection + Neo4j
```

---

## 6. Cross-cutting decisions

- **Storage.** Articles, entities, and relations live in **Weaviate** (per-use-case
  collections, created/owned by the orchestrators). MinHash signatures are stored as a Weaviate
  property; the FAISS + local-persistence article path is retired. **Neo4j** is written only at
  the end of pipeline 2. **Postgres** holds street geometry / edge ids for the geotagger.
- **`merge`** is an orchestrator operation, not a module. Article merge: append source+url, keep
  oldest date. Entity/relation merge: collapse a `dedup/corpus` cluster, then re-describe with
  `summarizer[aggregate]`.
- **Embeddings.** bge-m3 (1024-dim) everywhere via the existing `vectorizer`. Identical
  composition recipe at insert and query time per collection.
- **Statelessness.** Every module is stateless and independently testable; the orchestrators
  retain all intermediate output for a unit of work until its final upsert, so module contracts
  never thread state they don't own.

---

## 7. Open / deferred items

- **`ner` naming** — keep as `ner` or rename to the freed `extractor`. Non-blocking.
- **POI geocoding** — locations resolve via gazetteer only; arbitrary external geocoding is
  deferred (YAGNI).
- **Article re-embedding** — migrating to bge-m3 means existing MiniLM article vectors must be
  re-embedded during the Weaviate migration.

---

## 8. Deliverables

1. **This authoritative design spec.**
2. *Next phase:* per-module implementation documents rewritten over `docs/entity-pipeline/*`
   (`summarizer`, `nli`, `geotagger`, `ner`, `resolver`, `dedup`) plus **two new orchestrator
   specs** (news pipeline, relation pipeline). Each derives from this document.
