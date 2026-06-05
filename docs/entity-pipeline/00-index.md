# Entity Extraction Pipeline — Document Index

Two-pass, provenance-preserving pipeline for extracting entities and relations from
journalistic documents and integrating them into a knowledge graph.

## Pipeline order

```
Chunker (done)
    │  doc_id, char offsets
    ▼
Entity Extractor  [stateless: text + schema → entities, relations]
    │  entities[], relations[] with canonical types (verbatim mention_text, no offsets)
    ▼
Entity Resolver   [stateless, intra-doc: entities + relations → reduced entities + relations]
    │  canonical entities + relations rewritten to canonical names
    ▼
Entity Disambiguator  [per local entity]
    │  decision: merge | create | review
    ▼
Summarizer /entity    [on merge or create]
    │  refreshed description
    ▼
Persistence (future)
```

## The orchestrator

The NLP service exposes each stage as an independent, stateless endpoint. An **external
orchestrator** (the caller; out of scope for these docs) drives them in order and **retains
every intermediate output for a document until the final upsert**. This is the key to keeping
each service slim:

- It loads the ontology (01), projects it to an `ExtractionSchema`, chunks the document, and
  for each chunk calls the extractor with `(text, schema)`. It keeps every `ExtractResponse`
  (including each entity's full `attributes`) and tags it with that chunk's id and offsets —
  the extractor itself stays document-blind.
- It flattens the chunks' entities and relations (turning each extractor relation's chunk-local
  endpoint indices into the endpoint entity's name) and calls `/resolve`, which returns the
  reduced canonical entities and the relations rewritten to canonical names.
- It runs `/entity-disambiguate` per canonical entity, then `/summarize/entity` on merge/create.
- **At upsert it assembles the final record**: for each `ResolvedEntity` it gathers the source
  extracted entities behind its `names`, merges their `attributes` (non-null, highest-confidence
  wins), and writes the entity — core fields, Spanish description, and all schema-declared
  attributes — into Weaviate, plus the relations.

Because the orchestrator owns the assembled record, the resolver and disambiguator never
need to thread attributes through their contracts. Nothing is lost: data lives in the
orchestrator from extraction until it lands in the graph in one write.

## Documents

| # | File | Module | Status |
|---|------|--------|--------|
| 01 | [01-ontology.md](01-ontology.md) | `nlp/schema.py` | New |
| 02 | [02-entity-extractor.md](02-entity-extractor.md) | `nlp/entity_extractor/` | New |
| 03 | [03-normalizer.md](03-normalizer.md) | `nlp/resolver/` | New |
| 04 | [04-dedup-expansion.md](04-dedup-expansion.md) | `nlp/dedup/` | Expanded (existing unchanged) |
| 05 | [05-summarizer-expansion.md](05-summarizer-expansion.md) | `nlp/summarizer/` | Expanded (existing unchanged) |

## Key decisions recorded here

- **The extractor is a stateless function: `text + schema → entities, relations`.** It knows
  nothing about documents, chunks, offsets, IDs, or use-cases — the orchestrator owns all of
  that and passes the schema inline. This makes the extractor reusable for any schema and
  trivially testable.
- **Closed-world, discriminated-union grammar.** The extractor compiles the inline
  `ExtractionSchema` into a JSON-Schema grammar with one `oneOf` branch per type (a `const`
  discriminator) exposing *only that type's* subtypes and attributes — so the model can neither
  emit an undeclared type nor a foreign attribute nor a wrong datatype. No `__NOVEL__` /
  `__UNCLASSIFIED__` escape hatch; the grammar makes them unrepresentable. The one rule the
  grammar can't express — relation subject/object type compatibility — is the extractor's only
  post-check.
- **The LLM emits neither offsets nor names.** It copies `mention_text` / `evidence_text`
  verbatim and references relation endpoints by integer index into the entities array. No
  character spans are produced; the orchestrator locates those verbatim strings in its own text
  if it needs offsets. A leading `reasoning` scratchpad recovers chain-of-thought, which
  constrained decoding otherwise forbids.
- **The orchestrator owns the assembled record until upsert** (see above). Stages stay slim;
  attributes and edge confidence are joined back together at write time, not threaded through
  every contract.
- **The extractor and resolver never write prose — only the summarizer does.** Extraction
  and resolution copy verbatim text (`mention_text`, `evidence_text`) and decide structure;
  the only generated free-text in the pipeline is the canonical node description, written by
  the summarizer (05) **in Spanish**. One module owns prose, and the language is consistent.
- **Pronoun resolution runs upstream, on the whole document, before chunking** (02
  §preprocessing) — so a pronoun's antecedent (often in an earlier chunk) is resolved to its
  name and the extractor sees explicit subjects/objects. The extractor itself stays a pure
  per-chunk `text + schema` function.
- **The Weaviate entity collection is generated from the ontology**, including one property
  per declared entity attribute, so everything extracted has somewhere to land.
- **Resolution scales by pre-merge, then LLM refine.** A cheap pure-Python pass collapses
  exact-name repeats into candidates; the LLM only refines the reduced candidate set (definite
  descriptions, same-name splits), so the call stays small even for long documents. Relations
  are then rewritten to canonical names and overlap-deduplicated deterministically — no LLM, no
  aggregation.
- **The resolver is stateless and name-based, intra-doc.** In: a document's entities
  (name, type, evidence) + relations (endpoints by name). Out: reduced canonical entities +
  relations rewritten to canonical names. No `doc_id`, no ids, no offsets, no schema. Coreference
  (what classical NLP calls entity resolution) runs *after* extraction on the structured output,
  not as a pre-processing step on raw text.
- **Three resolution scopes, separated.** Pronouns → names (upstream, before chunking, 02);
  intra-document entity resolution (the resolver, 03); cross-document disambiguation against the
  graph (04). Only the last is a graph lookup; none of the three are the same step.
- **Relation type normalization done in extraction** via the grammar, not in a separate pass.
  The surface clause is preserved in `evidence_text` for audit; the `type` field holds the
  canonical relation type.
- **Article dedup untouched**. Entity disambiguation adds a Weaviate-backed path alongside
  the existing FAISS article path. No shared state or data structures.
- **One backward-compatible signature change** to `ollama_client.generate()` unlocks entity
  description maintenance without touching any existing caller.
