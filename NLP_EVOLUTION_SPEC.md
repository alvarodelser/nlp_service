# NLP Service — Evolution Spec

Status: draft. Sections marked **[TODO]** need decisions before implementation.

---

## 0. Conceptual answers behind the design

### Full text vs extractive summary for the relevance gate

The NLI gate uses a BERT model that truncates at 512 tokens (~380 words). For most articles
the model already silently discards the second half of the text. Passing the full article
is not more expensive — it's the same compute as passing the first 380 words.

However the truncation means the model only ever sees the opening of long articles, which
skews toward the lead. A cheap extractive pre-step (sentence scoring, already inside the
summariser) could give a more representative 200-word extract before the gate fires.
The efficient pipeline order is therefore:

```
raw text
  → [1] extractive extract   (cheap, ~0ms, always runs)
  → [2] relevance gate       (NLI or OOD, on the extract)
  → if OOS: discard
  → [3] LLM summarisation    (47s avg, only for in-scope articles)
  → [4] topic classification (on the LLM summary)
```

This avoids paying the LLM cost on discarded articles and gives the gate better signal.
The extractive step should be extracted from `nlp/summariser/extractive.py` into a
shared utility callable by both the gate and the summariser.

### Two embeddings per article

Every in-scope article gets two vectors, both from `paraphrase-multilingual-MiniLM-L12-v2`
(384-dim, L2-normalised). The model is loaded once and shared across all encode calls.

| Column | Input text | Generated at | Used for |
|--------|-----------|--------------|---------|
| `embedding_raw` *(name TBD)* | extractive extract (~200 words) | Step [2b], after extract | Dedup (FAISS), OOD relevance gate, topic proximity |
| `embedding_summary` *(name TBD)* | LLM summary (headline + body) | Step [6] after summarisation | Semantic search |

Column names will be revised to better reflect their purpose.

**`embedding_raw`** is computed from the **extractive extract**, not the full raw text.
Using the extract avoids the BERT 512-token truncation problem (raw text silently drops
the second half of long articles) and gives a more representative full-article signal.
The extractive step therefore runs before the embedding encode — a small reorder in the
pipeline (see step numbering below). This vector is written to both FAISS (dedup queries)
and `news.embedding_raw` (pgvector, for OOD and topic proximity).

**`embedding_summary`** is generated immediately after step [6] completes. Cost: a single
`model.encode()` call on ~60 words — negligible after the 47s LLM call that just ran.
Always generated for in-scope articles; never conditional.
OOS articles have no summary and therefore no `embedding_summary` (null).

The two vectors are intentionally different: `embedding_raw` captures a balanced
full-article domain signal (good for relevance and dedup); `embedding_summary` captures
the distilled semantic content (good for search — match a user query to what the article
is *about*).

---

## 1. Full ingestion pipeline — ordered steps

Each step is run by the ingestion orchestrator (not the NLP service itself, which exposes
independent endpoints). Steps are sequential; an article exits early if it fails a gate.

```
RAW ARTICLE (article_id, headline, raw_text, source, url, publication_date)
│
├─ [1] DEDUP — Part 1: MinHash LSH
│       Input : raw_text
│       Action: query MinHash index for Jaccard similarity to known articles
│       Output: duplicate_of (article_id) or None
│       If duplicate → ENRICH existing record (see below), then STOP.
│
│       ┌─ DUPLICATE ENRICHMENT (always runs on duplicate detection) ─────────────┐
│       │  UPDATE existing news row:                                               │
│       │    sources  → append {name, link, date} if source/url not already stored │
│       │    dates    → append publication_date if not already stored              │
│       │  This makes sources and dates arrays. Do NOT re-run any NLP steps.      │
│       └──────────────────────────────────────────────────────────────────────────┘
│
├─ [2a] EXTRACTIVE EXTRACT
│       Input : raw_text
│       Action: sentence scoring (TF-IDF salience, summariser/extractive.py)
│               select top sentences up to ~200 words
│       Output: extract (string)
│       Cost  : ~0ms. Always runs for unique articles.
│
├─ [2b] DEDUP — Part 2: Embedding similarity
│       Input : extract
│       Action: encode with paraphrase-multilingual-MiniLM-L12-v2 (384-dim, L2-normalised)
│               → query FAISS IndexFlatIP for cosine similarity
│       Output: duplicate_of (article_id) or None  |  embedding_raw vector (name TBD)
│       If duplicate → ENRICH existing record (same enrichment as step [1]), then STOP.
│       If unique    → add to MinHash + FAISS indexes
│                      persist embedding_raw to news.embedding_raw (pgvector)
│
│       ── Article is unique. Extract embedding stored. ──
│
├─ [3] RELEVANCE GATE (OOD centroid)
│       Input : embedding_raw (reused from [2b] — no re-encode)
│       Action: cosine similarity to stored in-scope centroid vector
│               centroid = mean of confirmed in-scope article embeddings (config artifact)
│       Output: relevance_score (float), in_scope (bool)
│       Threshold: relevance_threshold in config/topics.yaml
│       If OOS → store article with geo_scope=null, topics=[], out_of_scope=True. STOP.
│
│       ── Article is in scope. ──
│
├─ [4] LLM SUMMARISATION + SUMMARY EMBEDDING
│       Input : raw_text (extract passed as context to Ollama)
│       Action: Ollama LLM generates headline + 3-sentence summary
│               → encode summary with MiniLM → embedding_summary
│       Output: headline, summary, embedding_summary (384-dim)
│       Cost  : ~47s LLM + ~5ms encode. OOS articles never reach this step.
│
├─ [5] TOPIC CLASSIFICATION + SCOPE HYPOTHESIS
│       Input : summary
│       Model : Recognai/bert-base-spanish-wwm-cased-xnli (same NLI model, already loaded)
│       Action: single zero-shot call with combined label list:
│                 — topic labels from topics.yaml (multi-label)
│                 — scope hypotheses (treated as exclusive):
│                     "national": "Este artículo trata sobre una política o iniciativa de
│                                  movilidad a nivel nacional en España, sin estar limitado
│                                  a una ciudad o región específica."
│                     "regional": "Este artículo trata sobre movilidad en una comunidad
│                                  autónoma, provincia o región específica de España."
│               Scope label with highest score above threshold → initial geo_scope signal.
│               If neither scope hypothesis clears threshold → geo_scope undetermined here.
│       Output: topics[], scores{}, scope_signal (national | regional | null)
│       Cost  : one NLI call covering all labels. No extra model load.
│
└─ [6] GEOTAGGING  (runs last — GeoNames resolution is the slowest geo step)
        Input : raw_text + headline + scope_signal from [5]
        Action:
          Stage A — Toponym identification (text only, no lookup):
            • PlanTL-GOB-ES/roberta-base-bne-ner-capiter → LOC/GPE/FAC spans
            • Street prefix regex (Calle|Avenida|Plaza|Paseo|C\.|Avda\.) → hint="street"

          Stage B1 — City resolution:
            • For non-street spans: GeoNames lookup (feature_class P) → score candidates
            • cities_snapshot.json cross-reference → dominant city_id
            • Source prior (source_city_prior.json) breaks low-confidence ties
            • Street → city assignment: if hint="street" and city unknown,
                try span against OSM street index for all platform cities;
                unique match → city_id inferred from street index key.
                Ambiguous (same street name in multiple cities) → use source prior or
                leave city_id null on that street entry.

          Stage B2 — Street resolution (city-scoped):
            • For hint="street" spans with known city_id:
                normalise span (strip prefix, lowercase, strip accents)
                → lookup in street_index[city_id] → edge_ids[]
            • Unmatched street spans → GeoNames point lookup within city bbox

          Stage B3 — Regional / point resolution:
            • Remaining unresolved spans: GeoNames feature_class A → regional name
            • Any GeoNames lat/lon → geo_points entry

          Scope imputation (Pass 2):
            • If scope_signal == "national" → geo_scope = national (NLI wins)
            • If scope_signal == "regional" → geo_scope = regional
            • If scope_signal == null AND geo_cities non-empty → geo_scope = city
            • If scope_signal == null AND only regional matches → geo_scope = regional
            • If scope_signal == null AND no matches → geo_scope = national (fallback)

        Output: geo_scope, geo_region, geo_cities[], geo_streets[], geo_points[]

STORE: news row with headline, summary, topics, geo_scope, geo_region, geo_cities,
       geo_streets, geo_points, embedding_raw, embedding_summary, sources[], dates[]
```

### Why this order

- **Dedup first, but enrich not discard**: duplicates still carry new source/date
  information. MinHash fires first (cheapest); enrichment writes only if source/url differ.
- **Extract before embedding**: free step, gives the embedding a balanced full-article
  view instead of just the lead (BERT 512-token truncation otherwise silently drops half).
- **OOD gate reuses the dedup embedding**: encode once on the extract; same vector used
  for FAISS dedup and cosine distance to the in-scope centroid. No re-encode.
- **Summarise before classify**: topic NLI on the summary is more precise — compact,
  salient, fits in 512 tokens without truncation.
- **Scope hypothesis in the topic NLI call**: both use the same XNLI model already loaded.
  Adding 2 scope hypotheses costs ~2 extra forward passes; no extra model load.
  Scope signal from step [5] feeds into geotagging at step [6].
- **Geotagging last**: NER + GeoNames + street index lookup only runs for in-scope articles.
  The NLI scope signal from [5] is available as input, reducing scope imputation work.
- **Two embeddings, one model**: MiniLM loaded once. `embedding_raw` from extract at [2b],
  `embedding_summary` from LLM summary at [4]. Two encode calls per in-scope article.

---

## 2. Module changes

### 2.1 Relevance gate — migrate to OOD centroid

**Current**: NLI entailment of a hand-written hypothesis. False rejection rate = 0.57.

**Proposed**: Cosine similarity to an in-scope centroid built from known-good articles.

Steps:
1. Collect a reference set (~50–100 confirmed in-scope articles).
2. Encode with `paraphrase-multilingual-MiniLM-L12-v2`.
3. Store the centroid vector as a config artifact (numpy `.npy` file alongside `topics.yaml`).
4. At inference: compute cosine similarity to centroid. Threshold tunable via `config/topics.yaml`.
5. Keep NLI gate as a fallback option switchable by env var while OOD is being validated.

### 2.2 Topic taxonomy — rebuild from corpus

Run BERTopic (notebook `05_dataset_explore.ipynb`, section 7) on the full corpus.
Use discovered cluster keywords to rename or merge labels in `topics.yaml`.
Target: ≤ 15 tight, well-separated labels. Validate with per-label score distribution
(should be bimodal — articles are clearly in or out).

---

## 3. Geotagger — scope classification and extended resolution

### 3.0 Two explicit stages: identification then resolution

The geotagger must separate two conceptually distinct problems. Conflating them (current
state) makes city-first ordering impossible and obscures where errors come from.

**Stage A — Toponym identification**: find all place-referring spans in the text.
No external lookup. Output: `[{span, label, char_offset, hint}]`.
Tools:
- **PlanTL-GOB-ES/roberta-base-bne-ner-capiter** (replaces spaCy `es_core_news_lg`).
  RoBERTa fine-tuned on Spanish news corpus (BNE). Tags `LOC`, `PER`, `ORG`, `MISC`.
  Keep `LOC` spans; discard others.
- Rule-based street prefix regex (post-NER layer): `Calle|Avenida|Plaza|Paseo|C\.|Avda\.`
  followed by a capitalised name. Sets `hint = "street"` on the span, bypassing GeoNames
  and routing directly to the OSM street index in Stage B. Low noise — these prefixes are
  unambiguous in Spanish.

**Stage B — Toponym resolution**: for each identified span, find the real-world referent.
Run in two passes so city scope is known before street lookup:

```
Pass B1 — city resolution (scope-independent):
  For each span without hint="street":
    → GeoNames lookup (feature_class P) → score candidates (frequency, population, headline)
    → disambiguate to dominant city_id
    → also check feature_class A (admin division) for regional signals
  Source prior: use news source → city_prior mapping to break low-confidence ties

Pass B2 — sub-city resolution (city-scoped):
  With winning city_id known:
  For each span with hint="street" OR unresolved spans from B1:
    → OSM street index scoped to city_id (normalised name lookup)
    → if matched: return edge_ids list
    → if not matched: GeoNames point lookup within city bounding box
    → if still not matched: raw lat/lon from GeoNames (best available)
```

City-first is mandatory for street lookup: "Calle Mayor" exists in every Spanish city.
Without city scope, the street index is meaningless.

### 3.1 Scope model

Every article gets a **geo_scope** classification before place resolution:

| Scope | Meaning | Example |
|-------|---------|---------|
| `national` | Relevant to Spain broadly; no dominant city/region | "El Gobierno aprueba el Plan Estatal de Movilidad" |
| `regional` | Relevant to a comunidad autónoma or provincia | "La Junta de Andalucía subvencionará 200 km de carril bici" |
| `city` | Anchored to one or more specific cities | "Madrid amplía el carril bici de Gran Vía" |

NER for place identification runs **independently of scope** — it finds all mentioned
spans regardless of editorial focus. Scope is then determined in two passes:

**Pass 1 — explicit scope classifier** (NLI hypothesis or GeoNames feature class):
- If the dominant GeoNames match has `feature_class == A` (province, comunidad) → `regional`.
- If an explicit national-policy signal is found (NLI or keyword rule) → `national`.
- Otherwise → scope undetermined (null).

**Pass 2 — imputation from geotag results**:
- If scope is still null AND `geo_cities` is non-empty → impute `scope = city`.
  The presence of a resolved city is itself the evidence — no separate classifier needed.
- If scope is still null AND only regional GeoNames matches exist → impute `scope = regional`.
- If scope is still null AND no geo matches at all → `scope = national` (fallback:
  the article discusses mobility in Spain without naming a specific place).

This means scope is always set before storing. The imputation rule is the common case:
most city-specific articles don't need a classifier because the geotag already tells you.

An article can reference specific streets or points **within** a city-scoped article.
Scope describes the *editorial focus*, not the exhaustive list of places mentioned.

**Source prior**: news sources carry strong geographic signal. A `source_city_prior` mapping
(JSON config, e.g. `{"elmundo.es": null, "el-periodico.com": "Barcelona", ...}`) is used
in Pass B1 to break low-confidence ties and as an additional fallback in Pass 2 scope
imputation when geo matches are absent.

**[TODO]**: define the national-scope NLI hypothesis or keyword rule for Pass 1. Candidate:
*"Este artículo trata sobre política o regulación de movilidad a nivel nacional en España."*

**[TODO]**: build `source_city_prior.json` from the 421-article corpus source distribution
(already visible in `05_dataset_explore.ipynb` section 2).

### 3.2 Place resolution hierarchy

For every NER span the geotagger tries to resolve it, in order:

```
1. City match     → cities_snapshot.json name lookup (existing)
2. Street match   → edges.name fuzzy lookup (new, see 2.3)
3. Point match    → GeoNames lat/lon (existing), or geocoded coordinates
4. Regional match → GeoNames feature_class A (new)
5. Unresolved     → span text only, no coordinates
```

### 3.3 Street resolution against OSM edges

**Approach**: text-based name match against `edges.name` scoped to the resolved city.

At service startup, load a `street_index` per city: `{normalised_name: [edge_id, ...]}`.
Built from the same DB snapshot pattern as `cities_snapshot.json` — a JSON file generated
by a script that queries `edges` grouped by city_id and name.

At inference:
1. Normalise the NER span (lowercase, strip accents, strip "Calle/Avenida/..." prefix).
2. Look up in `street_index[city_id]`.
3. If match found → return `edge_id` list (one street name can span many OSM edges).

**Dummy for now**: the street_index file is a placeholder (empty dict). The real data
comes from a script `scripts/snapshot_streets.py` that will query the DB.
The geotagger returns `edge_ids` as a list; the backend does geometry queries using
`edges.geom` (PostGIS already indexed with GIST).

Street → city assignment: if a street is matched, the city is the one from the
`street_index` lookup. If the coordinates come from GeoNames without a street match,
assign city by checking `cities.bounds_*` bounding box containment.

### 3.4 Geotagger API response changes

```jsonc
{
  "geo_scope": "city",            // national | regional | city
  "geo_region": null,             // region name if scope == regional
  "geo_cities": [                 // list — can be >1 for regional/national scope
    { "city_id": 3, "city_name": "Madrid", "confidence": 0.71 }
  ],
  "geo_streets": [                // list — empty if no street resolved
    { "span": "Gran Vía", "edge_ids": [1021, 1022, 1023], "city_id": 3 }
  ],
  "geo_points": [                 // list — raw coordinates for unresolved places
    { "span": "Plaza de España", "lat": 40.423, "lon": -3.712, "geonames_id": 6359304 }
  ],
  "all_places": [...]             // existing field, unchanged
}
```

---

## 4. Database schema changes

### 4.1 `news` table additions

```sql
-- Sources and dates become arrays (enriched on duplicate detection)
-- sources replaces the scalar `source TEXT` and `link TEXT` columns
ALTER TABLE news
  ADD COLUMN sources  JSONB  DEFAULT '[]',  -- [{name, link, date}] — appended on each duplicate hit
  ADD COLUMN dates    JSONB  DEFAULT '[]';  -- [publication_date, ...] — one entry per source occurrence
-- Keep source TEXT and link TEXT for backward compat; deprecate after query migration.

-- Geo fields (replacing single `city TEXT`)
ALTER TABLE news
  ADD COLUMN geo_scope    TEXT,                           -- national | regional | city
  ADD COLUMN geo_region   TEXT,                           -- comunidad/provincia name
  ADD COLUMN geo_cities   JSONB   DEFAULT '[]',           -- [{city_id, city_name, confidence}]
  ADD COLUMN geo_streets  JSONB   DEFAULT '[]',           -- [{span, edge_ids, city_id}]
  ADD COLUMN geo_points   JSONB   DEFAULT '[]',           -- [{span, lat, lon, geonames_id}]

-- Two embedding columns (requires pgvector extension)
  ADD COLUMN embedding_raw     vector(384),   -- extract embed: dedup mirror, OOD, topic proximity
  ADD COLUMN embedding_summary vector(384);   -- summary embed: semantic search; null for OOS

-- Keep city TEXT for backward compat until queries are migrated, then drop
```

### 4.2 Vector indexes

```sql
CREATE EXTENSION IF NOT EXISTS vector;

-- OOD centroid queries and topic proximity: cosine on raw embedding
CREATE INDEX ON news USING ivfflat (embedding_raw vector_cosine_ops) WITH (lists = 50);

-- Semantic search: cosine on summary embedding (only non-null rows matter)
CREATE INDEX ON news USING ivfflat (embedding_summary vector_cosine_ops) WITH (lists = 50);
```

`embedding_raw` is written at step [2] (dedup), alongside the FAISS write.
`embedding_summary` is written at step [6] (summarisation), immediately after Ollama returns.
Both use the same MiniLM model instance — no additional model loading.
OOS articles have `embedding_raw` set and `embedding_summary` null.

---

## 5. Data requirements and testing strategy

### 5.1 Data files required by the geotagger

The NLP service never connects to the DB directly. All reference data is file-based,
loaded at startup. Three files are required:

| File | Script | Status |
|------|--------|--------|
| `nlp/geotagger/data/geonames_es.tsv` | `scripts/build_geonames_es.py` | Exists |
| `nlp/geotagger/data/cities_snapshot.json` | `scripts/snapshot_cities.py` | Exists |
| `nlp/geotagger/data/street_index.json` | `scripts/snapshot_streets.py` (to write) | **Missing** |

Street index shape: `{city_id: {normalised_name: [edge_id, ...]}}`.
Generated from `SELECT city_id, name, id FROM edges WHERE name IS NOT NULL ORDER BY city_id, name`.
Normalisation: lowercase, strip accents, strip leading street-type prefix (Calle/Avda/etc).
Path configurable via `STREET_INDEX_PATH` env var (same pattern as `TOPICS_YAML_PATH`).

### 5.2 Testing without the DB

Since all data files are static JSON/TSV, tests need only small fixture files — no DB,
no mocking framework required.

**`nlp/geotagger/data/cities_snapshot_test.json`** — 3–4 cities covering the fixture cases
(Madrid, Barcelona, Valladolid, Sevilla). Hand-written from the real cities table.

**`nlp/geotagger/data/street_index_test.json`** — 5–10 streets per test city, covering
the geo fixture article mentions:
```json
{
  "3": {
    "gran via": [1021, 1022, 1023],
    "paseo del prado": [2010, 2011],
    "calle alcala": [3005]
  }
}
```

Set `CITIES_SNAPSHOT_PATH` and `STREET_INDEX_PATH` env vars in tests to point to the
fixture files. Production paths remain the default.

**`nlp/geotagger/data/source_city_prior_test.json`** — handful of source → city mappings
covering the sources that appear in geo fixture cases.

### 5.3 Test layers

**Layer 1 — unit (offline, no service)**
Call `nlp.geotagger.service.run()` directly with fixture data files. Covers: NER span
extraction, city resolution, street lookup, scope imputation. Fast, no network.
Add to `eval/02_geotag_eval.ipynb` as new fixture cases for street and scope scenarios.

**Layer 2 — eval notebook (staging service)**
`eval/02_geotag_eval.ipynb` calls `/geotag` against the staging service with the full
(production) snapshot files. New fixture cases to add:
  - Street-prefix span (`C/ Gran Vía, Madrid`)
  - Headline-only city (geo-007, currently failing)
  - National-scope article (no city named)
  - Regional-scope article (comunidad autónoma named)

**Layer 3 — corpus run (real dataset, `05_dataset_explore.ipynb` section 6)**
After deployment, re-run section 6 on all 421 articles. Inspect: % with geo_scope resolved,
city distribution, street resolution rate, confidence distribution. Compare against the
pre-deployment baseline captured in the earlier notebook run.

---

## 6. Ingestion pipeline — orchestration

### 6.1 Existing flow

```
scripts/news_scrapper.py         — RSS scrape, word-overlap dedup, merge sources
  writes: data/news/movilidad_news_new.json   (new articles, no NLP)
           data/news/movilidad_news.json       (full archive)
```

The scraper's `merge_articles()` function already handles multi-source merging in the
JSON archive. The NLP pipeline adds DB persistence and deeper NLP.

### 6.2 New file: `ingestion/07_news/070_ingest_news.py`

Follows the existing numbering convention. Added to `run_ingestion.sh` as Phase 7.

```python
# Pseudocode — full implementation TBD
for article in read_new_articles("data/news/movilidad_news_new.json"):
    # [1] MinHash dedup
    result = POST("/dedup-check", {article_id, text})
    if result.duplicate_of:
        enrich_db_row(result.duplicate_of, article.sources)
        continue

    # [2a] Extractive extract (local import, not an endpoint)
    extract = extractive.summarise(article.raw_text, max_words=200)

    # [2b] Embedding dedup
    result = POST("/dedup-check-embed", {article_id, extract})
    if result.duplicate_of:
        enrich_db_row(result.duplicate_of, article.sources)
        continue
    embedding_raw = result.embedding   # returned by service, stored in pgvector

    # [3] Relevance gate (OOD)
    result = POST("/relevance", {article_id, embedding_raw})
    if not result.in_scope:
        insert_news_row(article, out_of_scope=True, embedding_raw=embedding_raw)
        continue

    # [4] Summarise + summary embedding
    result = POST("/summarize", {article_id, text, headline})
    headline, summary, embedding_summary = result.headline, result.summary, result.embedding

    # [5] Topics + scope signal
    result = POST("/classify", {article_id, text=summary})
    topics, scope_signal = result.topics, result.scope_signal

    # [6] Geotag
    result = POST("/geotag", {article_id, text, headline})
    geo = result   # geo_scope, geo_cities, geo_streets, geo_points

    # Store
    insert_news_row(article, headline, summary, topics, geo, embedding_raw, embedding_summary)
```

### 6.3 `run_ingestion.sh` addition

```bash
# 7. News ingestion (scrape + NLP pipeline)
echo -e "\n${GREEN}--- Phase 7: News Ingestion ---${NC}"
python3 scripts/news_scrapper.py          # fetch new articles → movilidad_news_new.json
python3 ingestion/07_news/070_ingest_news.py  # NLP pipeline → DB
```

---

## 7. Open questions / TODO

**NER**
- [ ] Replace `es_core_news_lg` with `PlanTL-GOB-ES/roberta-base-bne-ner-capiter` in `nlp/geotagger/ner.py`.
- [ ] Add street prefix regex layer after NER; validate on geo fixture cases.

**Data / snapshots**
- [ ] Write `scripts/snapshot_streets.py` querying `edges` table → `street_index.json`.
- [ ] Write `nlp/geotagger/data/street_index_test.json` fixture for unit tests.
- [ ] Build `source_city_prior.json` from corpus source distribution (notebook section 2).
- [ ] Add `CITIES_SNAPSHOT_PATH` and `STREET_INDEX_PATH` env var overrides to geotagger loader.

**Gazetteer / resolution**
- [ ] Refactor `geotagger/service.py`: explicit Stage A (identification) and Stage B (resolution).
- [ ] Implement Pass B1 → Pass B2 ordering (city-first, then street scoped to winning city).
- [ ] Add street prefix regex layer after PlanTL NER.

**Ingestion pipeline**
- [ ] Create `ingestion/07_news/070_ingest_news.py` following the pseudocode in section 6.2.
- [ ] Add Phase 7 to `run_ingestion.sh`.
- [ ] Decide whether extractive step is imported locally in 070 or called via a `/extract` endpoint.

**Scope**
- [ ] Add national + regional hypotheses to the topic NLI call in `nlp/classifier/service.py`.
- [ ] Add scope_signal output to `/classify` API response.
- [ ] Validate scope imputation rule on geo fixture cases (geo-007: headline-only city).

**Classifier / relevance**
- [ ] Decide on OOD threshold after running `05_dataset_explore.ipynb` section 7.
- [ ] Rebuild taxonomy from BERTopic clusters (section 7 of notebook).
- [ ] Decide whether topic classification runs on the LLM summary or the extractive extract.

**Schema**
- [ ] Confirm pgvector is installed on the DB host.
- [ ] Write migration adding `sources JSONB`, `dates JSONB`, geo columns, embedding columns.
- [ ] Update dedup service to write embedding_raw to DB alongside FAISS.
- [ ] Update dedup duplicate path to enrich sources/dates instead of bare STOP.
- [ ] Deprecation timeline for `news.city TEXT`, `news.source TEXT`, `news.link TEXT`.
