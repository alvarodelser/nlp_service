# Relation Pipeline — Orchestrator Implementation Doc

**Scope:** the relation-extraction orchestrator · **Status:** New (written from scratch)
**Authoritative source:** `docs/superpowers/specs/2026-06-05-nlp-two-pipelines-design.md` §5

## Purpose

A **synchronous Python orchestrator** that turns documents into a knowledge graph. It owns the
ontology (the `ExtractionSchema` it passes to `ner`), chunking, coref, Weaviate + Neo4j I/O,
cross-document `merge`, and step sequencing. The modules (`ner`, `resolver`, `summarizer`,
`dedup`) stay stateless and use-case-blind.

The flow has **two phases**: per-document ingestion, then a corpus-wide consolidation (the
corpus dedup is inherently batch).

```
PHASE A — ingest_document(doc, use_case)
  coref.resolve_pronouns(full text)  → chunk  → ner(chunk, schema) per chunk
        │  (flatten chunks: relation endpoint indices → entity names)
  resolver(entities, relations)              → reduced entities (ids) + relations (head_id/tail_id)
        │
  summarizer[entity_desc] per entity, [relation_desc] per relation   → descriptions
        │
  vectorizer.embed(description)              → vec (bge-m3)
        │
  upsert to per-use-case Weaviate entity/relation collections (with attributes)

PHASE B — consolidate(use_case)
  dedup.dedup_corpus(collection, entities)   → entity clusters
  dedup.dedup_corpus(collection, relations)  → relation clusters
        │
  for each cluster: merge() → summarizer[aggregate] → upsert FINAL Weaviate collection + Neo4j
```

---

## File layout

```
pipelines/
  relations/
    __init__.py
    orchestrator.py     ← ingest_document(...) and consolidate(...)
    ontology.py         ← load ontology YAML → ExtractionSchema + Weaviate collection spec
    chunker.py          ← split a document into chunks (or reuse the existing chunker)
    weaviate_io.py      ← per-doc + final collections, upsert, cluster member fetch, merge
    neo4j_io.py         ← write canonical nodes + relationships
config/
  ontologies/
    financial_flows.yaml
    <use_case>.yaml
```

Imports in-process: `from nlp.ner import service as ner, coref`,
`from nlp.ner.schema_types import ExtractionSchema, EntityTypeDef, ...`,
`from nlp.resolver import service as resolver`,
`from nlp.summarizer import service as summarizer`,
`from nlp.dedup import service as dedup`. `vectorizer` is the external bge-m3 service.

---

## Ontology — orchestrator-owned (`config/ontologies/*.yaml` + `ontology.py`)

There is no schema/ontology module in the NLP service; the orchestrator owns the type catalogue,
validates it, and emits two derived artefacts: the inline **`ExtractionSchema`** (doc 04) handed
to `ner`, and the **Weaviate collection spec** (one property per declared attribute).

```yaml
# config/ontologies/financial_flows.yaml
version: "1"
use_case: financial_flows
entity_types:
  - name: PERSON
    description: A natural person referenced by name, title, or role.
    subtypes:
      - {name: POLITICIAN, description: Elected or appointed public official.}
      - {name: EXECUTIVE,  description: Senior corporate officer.}
    attributes:
      nationality: {type: string, description: "ISO 3166-1 alpha-2 if known."}
      role_title:  {type: string, description: "Most specific stated role."}
  - name: ORGANIZATION
    description: Company, fund, foundation, or government body.
    subtypes: [{name: SHELL_COMPANY, description: No active operations.}]
    attributes:
      jurisdiction: {type: string, description: "Country of incorporation."}
relation_types:
  - name: PAYMENT_TO
    description: One party transfers money/assets to another.
    head_types: [PERSON, ORGANIZATION]
    tail_types: [PERSON, ORGANIZATION]
    attributes:
      amount:   {type: number, description: "Numeric amount, no symbol."}
      currency: {type: string, description: "ISO 4217 code."}
```

```python
# pipelines/relations/ontology.py
import os, yaml
from functools import lru_cache
from nlp.ner.schema_types import (ExtractionSchema, EntityTypeDef, RelationTypeDef,
                                  SubtypeDef, AttributeDef)

_DT = {"string": "text", "number": "number", "boolean": "boolean"}  # YAML datatype → Weaviate
_ONTOLOGY_DIR = os.environ.get("ONTOLOGY_DIR", "config/ontologies")

@lru_cache
def load(use_case: str) -> dict:
    data = yaml.safe_load(open(f"{_ONTOLOGY_DIR}/{use_case}.yaml"))
    _validate(data)                       # names SCREAMING_SNAKE; head/tail_types declared; datatypes valid
    return data

def extraction_schema(use_case: str) -> ExtractionSchema:
    data = load(use_case)
    return ExtractionSchema(
        entity_types=[EntityTypeDef(
            name=e["name"], description=e.get("description", ""),
            subtypes=[SubtypeDef(name=s["name"], description=s.get("description", ""))
                      for s in e.get("subtypes", [])],
            attributes=[AttributeDef(name=k, datatype=v["type"], description=v.get("description", ""))
                        for k, v in e.get("attributes", {}).items()],
        ) for e in data["entity_types"]],
        relation_types=[RelationTypeDef(
            name=r["name"], description=r.get("description", ""),
            head_types=r["head_types"], tail_types=r["tail_types"],
            attributes=[AttributeDef(name=k, datatype=v["type"], description=v.get("description", ""))
                        for k, v in r.get("attributes", {}).items()],
        ) for r in data["relation_types"]],
    )

def weaviate_entity_spec(use_case: str) -> dict:
    data = load(use_case)
    attr_props = []
    for e in data["entity_types"]:
        for k, v in e.get("attributes", {}).items():
            attr_props.append({"name": k, "dataType": [_DT[v["type"]]]})
    return {
        "name": f"Entity_{use_case}_v{data['version']}",
        "vectorizer_config": "none",
        "properties": [
            {"name": "canonical_name", "dataType": ["text"]},
            {"name": "type",           "dataType": ["text"]},
            {"name": "subtype",        "dataType": ["text"]},
            {"name": "description",    "dataType": ["text"]},      # Spanish (summarizer)
            {"name": "aliases",        "dataType": ["text[]"]},
            {"name": "doc_ids",        "dataType": ["text[]"]},
            {"name": "evidence",       "dataType": ["text[]"]},
            *_dedupe_by_name(attr_props),                          # one column per declared attribute
        ],
        "vectorIndexConfig": {"distance": "cosine"},
    }
```

A relation collection spec is built the same way (`Relation_{use_case}_v{n}` with `head_id`,
`tail_id`, `type`, `subtype`, `description`, `attributes`-JSON, `evidence`, `doc_ids`, `confidence`).
The collection name embeds use case + version so incompatible schemas never share a collection.

> This is where the deleted ontology doc's content lives now — orchestrator-owned, not a module.
> `ner` only ever sees the projected `ExtractionSchema`; it builds neither grammar source nor
> Weaviate spec.

---

## Phase A — `ingest_document`

```python
def ingest_document(doc: dict, use_case: str) -> dict:
    """doc: {doc_id, text, date?}. Extracts, resolves, describes, embeds, upserts per-doc."""
    schema = ontology.extraction_schema(use_case)
    wv = weaviate_io.Client(use_case)

    # 1. Coref on the WHOLE document, then chunk (antecedents cross chunks — doc 04 §0)
    text = coref.resolve_pronouns(doc["text"])
    chunks = chunker.split(text)

    # 2. ner per chunk; flatten, converting relation endpoint indices → entity names
    entities, relations = [], []
    for ch in chunks:
        r = ner.run(ch, schema)
        names = [e.name for e in r.entities]
        entities += [{"name": e.name, "type": e.type, "subtype": e.subtype,
                      "evidence": e.evidence_text} for e in r.entities]
        relations += [{"head": names[rel.head], "tail": names[rel.tail], "type": rel.type,
                       "subtype": rel.subtype, "evidence": rel.evidence_text,
                       "attributes": rel.attributes, "confidence": rel.confidence}
                      for rel in r.relations
                      if 0 <= rel.head < len(names) and 0 <= rel.tail < len(names)]

    # 3. Resolve intra-document (assigns ids; relations → head_id/tail_id)
    resolved = resolver.resolve(entities, relations)

    # 4. Describe each node/edge (summarizer profiles), 5. embed, 6. upsert
    id_map = {}
    for ent in resolved.entities:
        desc = summarizer.summarize("entity_desc", {
            "name": ent.canonical_name, "type": ent.type, "subtype": ent.subtype,
            "evidence": ent.evidence})["description"]
        vec = _embed(f"{ent.canonical_name}\n{desc}")
        id_map[ent.id] = wv.upsert_entity({
            "canonical_name": ent.canonical_name, "type": ent.type, "subtype": ent.subtype,
            "description": desc, "aliases": ent.names, "doc_ids": [doc["doc_id"]],
            "evidence": ent.evidence, **_entity_attributes(ent, resolved)}, vector=vec)

    for rel in resolved.relations:
        desc = summarizer.summarize("relation_desc", {
            "head": _name(resolved, rel.head_id), "tail": _name(resolved, rel.tail_id),
            "type": rel.type, "subtype": rel.subtype, "attributes": rel.attributes,
            "evidence": rel.evidence})["description"]
        vec = _embed(desc)
        wv.upsert_relation({
            "head_id": id_map[rel.head_id], "tail_id": id_map[rel.tail_id], "type": rel.type,
            "subtype": rel.subtype, "description": desc, "attributes": json.dumps(rel.attributes),
            "evidence": rel.evidence, "doc_ids": [doc["doc_id"]], "confidence": rel.confidence},
            vector=vec)

    return {"doc_id": doc["doc_id"], "entities": len(resolved.entities),
            "relations": len(resolved.relations)}
```

> **Attribute join at upsert.** Per `[[project_normalizer_edge_semantics]]` /
> `[[project_schema_variable]]`, the resolver never threads attributes. The orchestrator merges
> each `ResolvedEntity`'s source attributes (non-null, highest-confidence wins) into the declared
> attribute columns at upsert (`_entity_attributes`). Attributes are opaque pass-throughs — no
> module hardcodes domain rules.

---

## Phase B — `consolidate`

```python
def consolidate(use_case: str) -> dict:
    wv = weaviate_io.Client(use_case)
    final = weaviate_io.FinalClient(use_case)
    graph = neo4j_io.Graph(use_case)

    ent_clusters = dedup.dedup_corpus(wv.entity_collection, kind="entity",
                                      compare_property="description", type_filter="entity")["clusters"]
    rel_clusters = dedup.dedup_corpus(wv.relation_collection, kind="relation",
                                      compare_property="description", type_filter="relation")["clusters"]

    canon = {}
    for cluster in ent_clusters:
        members = [wv.get_entity(i) for i in cluster]
        merged = _merge_entity_cluster(members)                       # canonical name/type, union attrs, concat evidence
        agg = summarizer.summarize("aggregate", {
            "name": merged["canonical_name"], "type": merged["type"],
            "descriptions": [m["description"] for m in members],
            "evidence": merged["evidence"]})["description"]
        vec = _embed(f"{merged['canonical_name']}\n{agg}")
        cid = final.upsert_entity({**merged, "description": agg}, vector=vec)
        for i in cluster:
            canon[i] = cid
        graph.upsert_node(cid, merged | {"description": agg})

    for cluster in rel_clusters:
        members = [wv.get_relation(i) for i in cluster]
        merged = _merge_relation_cluster(members, canon)              # remap endpoints to canonical entity ids
        agg = summarizer.summarize("aggregate", {
            "name": f"{merged['type']}", "type": merged["type"],
            "descriptions": [m["description"] for m in members],
            "evidence": merged["evidence"]})["description"]
        vec = _embed(agg)
        rid = final.upsert_relation({**merged, "description": agg}, vector=vec)
        graph.upsert_edge(rid, merged | {"description": agg})

    return {"entities": len(ent_clusters), "relations": len(rel_clusters)}
```

### Cross-document `merge` (the orchestrator operation)

```python
def _merge_entity_cluster(members: list[dict]) -> dict:
    best = max(members, key=lambda m: len(m["canonical_name"]))       # most complete name
    attrs = {}
    for m in sorted(members, key=lambda m: m.get("_confidence", 0), reverse=True):
        for k, v in _attrs_of(m).items():
            if v is not None:
                attrs.setdefault(k, v)                                # highest-confidence non-null wins
    return {"canonical_name": best["canonical_name"], "type": best["type"],
            "subtype": best.get("subtype"),
            "aliases": sorted({a for m in members for a in m.get("aliases", [])}),
            "doc_ids": sorted({d for m in members for d in m.get("doc_ids", [])}),
            "evidence": [e for m in members for e in m.get("evidence", [])], **attrs}
```

Relation clusters merge the same way (max confidence, union attributes, concat evidence),
remapping `head_id`/`tail_id` through `canon` (the per-document → canonical entity id map built in
the entity loop). **Relations are never aggregated across distinct instances** — `dedup_corpus`
only clusters relations the LLM/threshold judged the *same* relation
(`[[project_normalizer_edge_semantics]]`).

---

## Neo4j (`pipelines/relations/neo4j_io.py`)

Written **only** in Phase B. Canonical entities → nodes (label = `type`, props = canonical_name,
description, attributes); canonical relations → edges (type = relation type, props = description,
attributes, confidence). Keyed by the final Weaviate id so the graph and vector store stay
joinable.

```python
class Graph:
    def upsert_node(self, cid, props): ...   # MERGE (n {id:$cid}) SET n += $props, n:`<type>`
    def upsert_edge(self, rid, props): ...   # MATCH endpoints by canonical id; MERGE edge; SET props
```

---

## Error handling

| Failure | Behaviour |
|---|---|
| `ner`/`resolver`/`summarizer` Ollama down | Abort the current document; nothing upserted for it. Re-run is idempotent on `doc_id`. |
| `vectorizer` down | Abort before the upsert that needs the vector. |
| Weaviate down | Abort; Phase A upserts per node/edge, so a partial document is possible — re-running with the same `doc_id` overwrites (idempotent upsert keyed by content/doc_id). |
| Neo4j down (Phase B) | The Weaviate final upsert still succeeds; log and let Phase B be re-runnable to backfill the graph. |

Phase B is **idempotent and re-runnable**: it reads the current collections and rebuilds
clusters, so a failed consolidation can simply be re-invoked.

---

## Configuration (env)

| Env var | Default | Description |
|---|---|---|
| `ONTOLOGY_DIR` | `config/ontologies` | Ontology YAML directory |
| `VECTORIZER_URL` | `http://vectorizer:8000/embed` | bge-m3 embedding service |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | — | Neo4j connection |
| `WEAVIATE_URL` | `http://weaviate:8080` | Weaviate HTTP (REST + GraphQL via httpx; shared) |
| `RELATIONS_CHUNK_TOKENS` | `512` | Chunk size for `chunker.split` |

Module env (extraction/resolve/summarize models, dedup corpus thresholds) is owned by docs 04–06.

---

## Dependencies

- Modules: `nlp.ner`, `nlp.resolver`, `nlp.summarizer`, `nlp.dedup` (docs 04, 05, 01, 06).
- `httpx` (Weaviate REST + GraphQL — no `weaviate-client`), `neo4j` (**new package**), `pyyaml`.
- External `vectorizer` (bge-m3) + Ollama sidecar.
