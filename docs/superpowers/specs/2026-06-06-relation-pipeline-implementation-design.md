# Relation Pipeline — Implementation Design

**Status:** Approved design, ready for TDD implementation. Date: 2026-06-06.
**Derives from:** `docs/superpowers/specs/2026-06-05-nlp-two-pipelines-design.md` §5 and
`docs/entity-pipeline/08-relation-pipeline.md`. Where doc 08 disagrees with the actual module
code, this document records the reconciled truth and wins.

## Purpose

A synchronous Python library — `pipelines/relations/` — that turns documents into a knowledge
graph. It owns the ontology (the `ExtractionSchema` handed to `ner`), chunking, coref ordering,
Weaviate + Neo4j I/O, cross-document `merge`, and step sequencing. The `nlp` modules (`ner`,
`resolver`, `summarizer`, `dedup`) stay stateless and use-case-blind. Two entry points:
`ingest_document(doc, use_case)` and `consolidate(use_case)`.

## Flow

```
PHASE A — ingest_document(doc, use_case)
  coref.resolve_pronouns(full text) → chunker.split → ner.run(chunk, schema) per chunk
    flatten chunks (relation endpoint indices → entity names)
  resolver.resolve(entities, relations) → ResolvedEntity (ids) + ResolvedRelation (head_id/tail_id)
  summarizer[entity_desc] / [relation_desc] → descriptions ; vectorizer.embed → vec
  upsert to working Entity_/Relation_ collections (attributes joined from ner)

PHASE B — consolidate(use_case)
  dedup.dedup_corpus(entity_collection)  → entity clusters
  dedup.dedup_corpus(relation_collection) → relation clusters
  per cluster: merge() → summarizer[aggregate] → embed → upsert FINAL Weaviate + Neo4j (Phase B only)
```

## File layout

```
pipelines/
  relations/
    __init__.py        ← exports ingest_document, consolidate
    ontology.py        ← load() + extraction_schema() + weaviate_entity_spec()/weaviate_relation_spec()
    chunker.py         ← split(text): word-budget splitter (RELATIONS_CHUNK_TOKENS)
    attributes.py      ← name→(attributes,confidence) map + join onto ResolvedEntity
    weaviate_io.py     ← Client (working collections) + FinalClient; ensure/upsert/get
    merge.py           ← _merge_entity_cluster, _merge_relation_cluster
    neo4j_io.py        ← Graph: upsert_node, upsert_edge (lazy neo4j import, injectable driver)
    orchestrator.py    ← ingest_document, consolidate, _embed, small helpers
config/
  ontologies/
    financial_flows.yaml
```

In-process imports: `from nlp.ner import service as ner, coref`,
`from nlp.ner.schema_types import ExtractionSchema, EntityTypeDef, RelationTypeDef, SubtypeDef,
AttributeDef`, `from nlp.resolver import service as resolver`,
`from nlp.summarizer import service as summarizer`, `from nlp.dedup import service as dedup`. The
vectorizer is the external bge-m3 HTTP service.

## Reconciliations with doc 08 (verified against module code)

1. **Attribute join — the resolver drops attributes.** `EntityIn` and `ResolvedEntity` carry no
   `attributes`, so `resolver.resolve` cannot thread them. Doc 08's `_entity_attributes(ent,
   resolved)` therefore has no data source. The orchestrator instead keeps its own
   `name → (attributes, confidence)` map built from the `ner` `ExtractedEntity` list (which does
   carry `attributes` and `confidence`), and for each `ResolvedEntity` unions attributes across
   its `names` with highest-confidence-non-null-wins. This lives in `attributes.py`. Honors
   `[[project_schema_variable]]` / `[[project_normalizer_edge_semantics]]` — attributes are opaque
   pass-throughs, no module hardcodes domain rules.
2. **`dedup_corpus` type_filter.** Doc passes `type_filter="entity"/"relation"`, but the stored
   `type` property holds entity types (`PERSON`, …). Entities and relations live in separate
   collections, so `consolidate` passes `type_filter=None` and clusters each collection whole.
3. **Chunker.** No chunker exists in `nlp`. Build `chunker.split(text)` — a paragraph-aware
   word-budget splitter (`RELATIONS_CHUNK_TOKENS`, default 512, words as a token proxy, matching
   the summarizer's approach).
4. **Entity `confidence` column.** Added to the entity collection (doc omits it) so Phase-B
   `_merge_entity_cluster` can order attributes by confidence. Set in Phase A to the max member
   confidence.
5. Verified to match doc 08 as-is: `coref.resolve_pronouns(text) -> str`;
   `ner.run(chunk, schema) -> NerResponse(entities=[ExtractedEntity], relations=[ExtractedRelation])`
   where relation endpoints are integer indices; `resolver.resolve(entities, relations)` accepting
   plain dicts and returning `ResolveResponse`; summarizer profiles `entity_desc`
   (`{name,type,subtype,evidence[]}`), `relation_desc` (`{head,tail,type,subtype,attributes,
   evidence[]}`), `aggregate` (`{name,type,descriptions[],evidence[]}`); `dedup.dedup_corpus(
   collection, kind, compare_property, type_filter) -> {"clusters": [[id,...], ...]}`.

## Ontology (ontology.py + config/ontologies/*.yaml)

`load(use_case)` reads `${ONTOLOGY_DIR}/{use_case}.yaml`, validates (names SCREAMING_SNAKE,
relation `head_types`/`tail_types` declared and referencing known entity types, datatypes in
{string,number,boolean}). `extraction_schema(use_case)` projects an `ExtractionSchema` for `ner`.
`weaviate_entity_spec`/`weaviate_relation_spec` emit Weaviate class specs with one column per
declared attribute. Collection names: `Entity_{uc}_v{version}`, `Relation_{uc}_v{version}`, and
`Entity_{uc}_final_v{version}` / `Relation_{uc}_final_v{version}`.

Entity columns: `canonical_name`, `type`, `subtype`, `description`, `aliases[]`, `doc_ids[]`,
`evidence[]`, `confidence`, + one per entity attribute.
Relation columns: `head_id`, `tail_id`, `type`, `subtype`, `description`, `attributes` (JSON text),
`evidence[]`, `doc_ids[]`, `confidence`, + one per relation attribute.

Ship `financial_flows.yaml` (PERSON/ORGANIZATION + PAYMENT_TO) as the documented transparency
use case.

## Phase A — ingest_document

1. `coref.resolve_pronouns(doc["text"])` on the whole document, then `chunker.split`.
2. `ner.run(chunk, schema)` per chunk; flatten — entities → `{name,type,subtype,evidence}` dicts
   for the resolver, and the parallel `name → (attributes, confidence)` map; relations →
   `{head:name, tail:name, type, subtype, evidence, attributes, confidence}` (endpoint index →
   name; ner already drops out-of-range/type-incompatible relations, the orchestrator re-guards).
3. `resolver.resolve(entities, relations)` → ids + rewritten relations.
4. Per entity: `summarizer.summarize("entity_desc", …)`, `_embed`, `wv.upsert_entity(props +
   joined attributes + max confidence, vector)`; build `id_map[resolved.id] → weaviate id`.
5. Per relation: `summarizer.summarize("relation_desc", …)`, `_embed`, `wv.upsert_relation(props
   with head_id/tail_id remapped through id_map, attributes JSON, vector)`.
Returns `{doc_id, entities, relations}`.

## Phase B — consolidate

`dedup_corpus` for entities and relations (`type_filter=None`). For each entity cluster: fetch
members, `_merge_entity_cluster`, `aggregate` summarize, embed, `final.upsert_entity`, record
`canon[member_id] → final id`, `graph.upsert_node`. For each relation cluster: fetch members,
`_merge_relation_cluster` (remap `head_id`/`tail_id` through `canon`), aggregate, embed,
`final.upsert_relation`, `graph.upsert_edge`. Returns `{entities, relations}` (cluster counts).

### merge (merge.py)

`_merge_entity_cluster(members, attr_names)` — most-complete (longest) canonical name + its type;
union aliases, doc_ids; concat evidence; attributes union with highest-`confidence` non-null wins
(restricted to `attr_names`). `_merge_relation_cluster(members, attr_names, canon)` — same union,
remap endpoints through `canon`, parse each member's `attributes` JSON. Relations are never
aggregated across distinct instances — `dedup_corpus` only clusters relations judged the same
(`[[project_normalizer_edge_semantics]]`).

## Neo4j (neo4j_io.py)

`Graph(use_case, driver=None)` lazy-imports `neo4j` and builds a driver from `NEO4J_URI` /
`NEO4J_USER` / `NEO4J_PASSWORD` when none injected (tests inject a mock). `upsert_node(cid, props)`
→ `MERGE (n {id:$cid}) SET n += $props` with label = entity type. `upsert_edge(rid, head_id,
tail_id, type, props)` → match endpoints by canonical id, `MERGE` a typed edge, `SET` props.
Written only in Phase B; keyed by the final Weaviate id.

## Error handling

| Failure | Behaviour |
|---|---|
| `ner`/`resolver`/`summarizer` Ollama down | Abort current document; nothing finalised for it. Re-run idempotent on `doc_id`. |
| `vectorizer` down | Abort before the upsert that needs the vector. |
| Weaviate down | Abort; re-running the same `doc_id` overwrites (idempotent). |
| Neo4j down (Phase B) | Final Weaviate upsert still succeeds; log and let Phase B re-run backfill the graph. |

Phase B is idempotent and re-runnable: it reads current collections and rebuilds clusters.

## Env config

`ONTOLOGY_DIR` (`config/ontologies`), `VECTORIZER_URL` (`http://vectorizer:8000/embed`),
`RELATIONS_CHUNK_TOKENS` (`512`), `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD`,
`WEAVIATE_URL` (`http://weaviate:8080`). Module env (extraction/resolve/summarize models, dedup
corpus thresholds) stays owned by the modules.

## Dependencies

Modules `nlp.ner`, `nlp.resolver`, `nlp.summarizer`, `nlp.dedup`; `httpx` + `pyyaml`; **new:
`neo4j` driver**; external `vectorizer` (bge-m3) + Ollama sidecar.

## Testing (TDD, mocked deps)

Tests written before implementation, mocking `ner`/`coref`/`resolver`/`summarizer`/`dedup`,
`weaviate_io` clients, `neo4j` driver, and the vectorizer HTTP call. Coverage: ontology
load/validation/projection + specs; chunker; attribute-join (union, confidence ordering,
unresolved names); Phase A ingest (chunk fan-out, endpoint remap, attribute join at upsert);
Phase B consolidate (cluster → merge → aggregate → final upsert → graph, endpoint remap through
canon); both merge helpers; `weaviate_io` (httpx MockTransport); `neo4j_io` (mock driver/session).
No live Ollama / Weaviate / Neo4j / b4c.
