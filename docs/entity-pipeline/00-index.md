# Entity Extraction Pipeline — Document Index

Two-pass, provenance-preserving pipeline for extracting entities and relations from
journalistic documents and integrating them into a knowledge graph.

## Pipeline order

```
Chunker (done)
    │  doc_id, char offsets, text
    ▼
Entity Extractor  [Pass 1, per chunk]   POST /entity-extract
    │  entities[], relations[] with canonical types, spans, attributes
    ▼
Intra-Doc Clusterer  [per document]     POST /cluster
    │  LocalEntity[], LocalEdge[] with local_ids and absolute offsets
    ▼
Entity Disambiguator  [per LocalEntity]  POST /entity-disambiguate
    │  decision: merge | create | review
    ▼
Summarizer /entity    [on merge or create]  POST /summarize/entity
    │  refreshed canonical description
    ▼
Persistence (out of scope)
```

Article dedup runs in parallel at ingestion time:

```
Chunker
    │  article_id + pre-computed embedding
    ▼
Article Dedup   POST /dedup
    │  duplicate_of | indexed
    ▼
(continue pipeline or discard)
```

## Documents

| # | File | Module | Status |
|---|------|--------|--------|
| 01 | [01-ontology.md](01-ontology.md) | `nlp/schema.py` | New |
| 02 | [02-entity-extractor.md](02-entity-extractor.md) | `nlp/entity_extractor/` | New |
| 03 | [03-normalizer.md](03-normalizer.md) | `nlp/clusterer/` | New |
| 04 | [04-dedup-expansion.md](04-dedup-expansion.md) | `nlp/dedup/` | Rewritten (stateless, Weaviate) |
| 05 | [05-summarizer-expansion.md](05-summarizer-expansion.md) | `nlp/summarizer/` | Expanded (existing unchanged) |

## Key decisions

- **Closed schema (Phase 0).** Entity and relation types are declared in YAML before any
  document is processed. The LLM extracts only from those types; no escape hatches.

- **Three mechanical translations from schema YAML.** JSON Schema for Ollama constrained
  decoding (structural), system prompt for per-type attribute requirements (semantic),
  Weaviate collection spec for persistence. No intelligence — pure loops over the parsed YAML.

- **Free `attributes` object in extraction output.** The JSON Schema passes
  `{"type": "object"}` for attributes rather than a union of all attribute keys across all
  types. System prompt carries per-type requirements; application code validates required
  fields post-extraction.

- **Two LLM passes per document.** Pass 1 extracts per chunk (entities + relations with
  canonical types via enum). Pass 2 clusters entity mentions within the document. Cross-doc
  disambiguation is a third step with no LLM call for high-confidence cases.

- **No edge merging.** Multiple edges between the same entity pair are valid distinct
  events. No amount summing, no attribute merging, no deduplication of relations.

- **Pronoun and role resolution in the clusterer.** A second constrained LLM call resolves
  pronouns and role titles against the named entities already found in the same document.
  Step is skipped entirely if no unresolved mentions are detected.

- **Stateless dedup.** Both article dedup and entity disambiguation query Weaviate
  per-request. No in-process MinHash, FAISS, or global indexes. `minhash_index.py`,
  `embedding_index.py`, `id_map.py`, and `persistence.py` are removed.

- **Same dedup pattern for articles and entities.** Caller sends a pre-computed embedding;
  service queries Weaviate; returns a decision. Thresholds are env-var-configured in the
  service, not in the caller.

- **One backward-compatible signature change** to `ollama_client.generate()` unlocks entity
  description maintenance without touching any existing caller.
