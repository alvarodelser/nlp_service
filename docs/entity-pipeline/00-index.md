# Entity Extraction Pipeline — Document Index

Two-pass, provenance-preserving pipeline for extracting entities and relations from
journalistic documents and integrating them into a knowledge graph.

## Pipeline order

```
Chunker (done)
    │  doc_id, char offsets
    ▼
Entity Extractor  [Pass 1, per chunk]
    │  entities[], relations[] with canonical types and spans
    ▼
Normalizer        [Pass 2, per document]
    │  entity_clusters (local_id), merged edges
    ▼
Entity Disambiguator  [per local entity]
    │  decision: merge | create | review
    ▼
Summarizer /entity    [on merge or create]
    │  refreshed description
    ▼
Persistence (future)
```

## Documents

| # | File | Module | Status |
|---|------|--------|--------|
| 01 | [01-ontology.md](01-ontology.md) | `nlp/ontology.py` | New |
| 02 | [02-entity-extractor.md](02-entity-extractor.md) | `nlp/entity_extractor/` | New |
| 03 | [03-normalizer.md](03-normalizer.md) | `nlp/normalizer/` | New |
| 04 | [04-dedup-expansion.md](04-dedup-expansion.md) | `nlp/dedup/` | Expanded (existing unchanged) |
| 05 | [05-summarizer-expansion.md](05-summarizer-expansion.md) | `nlp/summarizer/` | Expanded (existing unchanged) |

## Key decisions recorded here

- **Flat enums in JSON Schema** for Ollama constrained decoding; hierarchy lives in the
  system prompt and in YAML config. Prevents grammar compilation overhead and malformed
  outputs from deeply nested schemas.
- **Two passes per document**, not one or three. Pass 1 extracts per chunk (entities +
  relations with canonical types via enum). Pass 2 clusters entity mentions and deduplicates
  edges for the whole document. Cross-doc disambiguation is a third separate step, not an
  LLM pass.
- **Intra-doc resolver merged into normalizer**. Entity clustering (what classical NLP
  calls coreference resolution) runs *after* extraction on the structured output, not as a
  pre-processing step on raw text. The structured mention list gives the LLM cleaner input
  and preserves span provenance.
- **Relation type normalization done in Pass 1** via enum, not in a separate pass. Surface
  phrases are preserved in `description`/`evidence_span` for audit; the `relation` field
  holds the canonical type.
- **Article dedup untouched**. Entity disambiguation adds a Weaviate-backed path alongside
  the existing FAISS article path. No shared state or data structures.
- **One backward-compatible signature change** to `ollama_client.generate()` unlocks entity
  description maintenance without touching any existing caller.
