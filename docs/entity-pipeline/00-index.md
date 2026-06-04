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
Entity Disambiguator  [per LocalEntity]  POST /dedup  (type: "entity")
    │  decision: merge | create | review
    ▼
Summarizer            [on merge or create]  POST /summarize  (type: "entity" | "relation")
    │  refreshed canonical description
    ▼
Persistence (out of scope)
```

Article dedup runs in parallel at ingestion time:

```
Chunker
    │  article_id + text + optional embedding
    ▼
Article Dedup   POST /dedup  (type: "article")
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

- **`[PRONOUN]` token for unresolved pronouns.** If the extractor finds a pronoun or bare
  role title with no named antecedent in the same chunk, it uses `[PRONOUN]` as the entity
  name (confidence 0.3) and preserves the relation and its span. The clusterer resolves
  these against the full document's named entity list in the same combined LLM call —
  `[PRONOUN]` mentions are annotated `NEEDS_RESOLUTION` in the prompt.

- **Jaccard name hints in clusterer prompt.** Before the LLM call, word-token Jaccard is
  computed for all mention pairs. Pairs ≥ 0.5 are annotated `LIKELY_SAME` in the prompt;
  `[PRONOUN]` mentions are annotated `NEEDS_RESOLUTION`. No rule-based resolution — the LLM
  makes all decisions using these annotations as hints.

- **Combined clustering + edge wiring in one LLM call.** The clusterer prompt includes both
  the flat entity mention list and the flat relation list. The LLM returns clusters with
  `mention_indices` AND edges with `head_local_id` / `tail_local_id` assigned in the same
  output. No second rule-based pass.

- **MinHash kept; FAISS removed.** Article dedup: Stage 1 MinHash (in-process, file-backed
  pickle at `DEDUP_DATA_DIR/{use_case}/minhash_lsh.pkl`) catches exact/near-exact reprints
  cheaply. Stage 2 Weaviate handles semantic duplicates. `embedding_index.py` and `id_map.py`
  deleted; `faiss-cpu` removed. Entity disambiguation uses Weaviate only.

- **Unified `POST /dedup`.** Article dedup and entity disambiguation share one endpoint,
  dispatched by a `type` discriminator (`"article"` | `"entity"`). Same pattern: caller
  sends pre-computed embedding, service queries Weaviate, returns decision.

- **Unified `POST /summarize`.** Article rewriting, entity description, and relation
  description share one endpoint, dispatched by `type` (`"article"` | `"entity"` |
  `"relation"`). Prompts live in the service. Extensible to further types without changing
  existing paths.
