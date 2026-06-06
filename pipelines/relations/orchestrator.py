"""The relation-extraction orchestrator. Two phases: per-document ingestion (ingest_document) and
corpus-wide consolidation (consolidate). It owns the ontology, chunking, coref ordering, Weaviate
+ Neo4j I/O, cross-document merge, and step sequencing. The nlp modules stay stateless and
use-case-blind."""
import json
import os

import httpx

from nlp.ner import service as ner, coref
from nlp.resolver import service as resolver
from nlp.summarizer import service as summarizer
from nlp.dedup import service as dedup

from . import ontology, chunker, weaviate_io, neo4j_io, merge as merge_mod
from .attributes import build_attribute_index, join_entity_attributes

_VECTORIZER_URL = os.environ.get("VECTORIZER_URL", "http://vectorizer:8000/embed")
_VECTORIZER_TIMEOUT = float(os.environ.get("VECTORIZER_TIMEOUT", "30"))


def _embed(text: str) -> list[float]:
    r = httpx.post(_VECTORIZER_URL, json={"text": text}, timeout=_VECTORIZER_TIMEOUT)
    r.raise_for_status()
    return r.json()["embedding"]


def _name(resolved, entity_id: str) -> str:
    for e in resolved.entities:
        if e.id == entity_id:
            return e.canonical_name
    return entity_id


def _entity_confidence(names, attr_index) -> float:
    confs = [conf for n in names for conf, _attrs in attr_index.get(n, [])]
    return max(confs, default=0.0)


# --- Phase A ---

def ingest_document(doc: dict, use_case: str) -> dict:
    """doc: {doc_id, text, date?}. Extracts, resolves, describes, embeds, upserts per-document."""
    schema = ontology.extraction_schema(use_case)
    attr_names = ontology.entity_attribute_names(use_case)
    wv = weaviate_io.Client(use_case)
    stage = "ensure"
    try:
        wv.ensure_collections()

        # 1. Coref on the whole document, then chunk (antecedents cross chunks).
        stage = "coref"
        text = coref.resolve_pronouns(doc["text"])
        chunks = chunker.split(text)

        # 2. ner per chunk; flatten (relation endpoint indices → entity names) and keep a parallel
        #    name → (attributes, confidence) record set for the attribute join.
        stage = "extract"
        entities, relations, attr_records = [], [], []
        for ch in chunks:
            r = ner.run(ch, schema)
            names = [e.name for e in r.entities]
            for e in r.entities:
                entities.append({"name": e.name, "type": e.type, "subtype": e.subtype,
                                 "evidence": e.evidence_text})
                attr_records.append({"name": e.name, "attributes": e.attributes,
                                     "confidence": e.confidence})
            for rel in r.relations:
                if 0 <= rel.head < len(names) and 0 <= rel.tail < len(names):
                    relations.append({"head": names[rel.head], "tail": names[rel.tail],
                                      "type": rel.type, "subtype": rel.subtype,
                                      "evidence": rel.evidence_text, "attributes": rel.attributes,
                                      "confidence": rel.confidence})

        # 3. Resolve intra-document (assigns ids; relations → head_id/tail_id).
        stage = "resolve"
        resolved = resolver.resolve(entities, relations)
        attr_index = build_attribute_index(attr_records)

        # 4-6. Describe, embed, upsert each node then edge.
        stage = "entities"
        id_map: dict[str, str] = {}
        for ent in resolved.entities:
            desc = summarizer.summarize("entity_desc", {
                "name": ent.canonical_name, "type": ent.type, "subtype": ent.subtype,
                "evidence": ent.evidence})["description"]
            vec = _embed(f"{ent.canonical_name}\n{desc}")
            joined = join_entity_attributes(ent.names, attr_index, attr_names)
            id_map[ent.id] = wv.upsert_entity({
                "canonical_name": ent.canonical_name, "type": ent.type, "subtype": ent.subtype,
                "description": desc, "aliases": ent.names, "doc_ids": [doc["doc_id"]],
                "evidence": ent.evidence, "confidence": _entity_confidence(ent.names, attr_index),
                **joined}, vector=vec)

        stage = "relations"
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

    except (httpx.HTTPError, RuntimeError):
        return {"doc_id": doc.get("doc_id"), "status": "error", "stage": stage}


# --- Phase B ---

def _graph_node_props(node_props: dict) -> dict:
    # Neo4j props must be primitives or arrays of primitives; drop the large evidence list.
    return {k: v for k, v in node_props.items() if k != "evidence"}


def _graph_edge_props(rel_props: dict) -> dict:
    return {k: v for k, v in rel_props.items()
            if k not in ("head_id", "tail_id", "type", "evidence")}


def consolidate(use_case: str) -> dict:
    wv = weaviate_io.Client(use_case)
    final = weaviate_io.FinalClient(use_case)
    final.ensure_collections()
    graph = neo4j_io.Graph(use_case)
    ent_attr_names = ontology.entity_attribute_names(use_case)
    rel_attr_names = ontology.relation_attribute_names(use_case)

    ent_clusters = dedup.dedup_corpus(wv.entity_collection, kind="entity",
                                      compare_property="description")["clusters"]
    rel_clusters = dedup.dedup_corpus(wv.relation_collection, kind="relation",
                                      compare_property="description")["clusters"]

    canon: dict[str, str] = {}
    for cluster in ent_clusters:
        members = [wv.get_entity(i) for i in cluster]
        merged = merge_mod._merge_entity_cluster(members, ent_attr_names)
        agg = summarizer.summarize("aggregate", {
            "name": merged["canonical_name"], "type": merged["type"],
            "descriptions": [m.get("description", "") for m in members],
            "evidence": merged["evidence"]})["description"]
        vec = _embed(f"{merged['canonical_name']}\n{agg}")
        node_props = {**merged, "description": agg}
        cid = final.upsert_entity(node_props, vector=vec)
        for i in cluster:
            canon[i] = cid
        graph.upsert_node(cid, _graph_node_props(node_props))

    for cluster in rel_clusters:
        members = [wv.get_relation(i) for i in cluster]
        merged = merge_mod._merge_relation_cluster(members, rel_attr_names, canon)
        agg = summarizer.summarize("aggregate", {
            "name": merged["type"], "type": merged["type"],
            "descriptions": [m.get("description", "") for m in members],
            "evidence": merged["evidence"]})["description"]
        vec = _embed(agg)
        rel_props = {**merged, "description": agg, "attributes": json.dumps(merged["attributes"])}
        rid = final.upsert_relation(rel_props, vector=vec)
        graph.upsert_edge(rid, merged["head_id"], merged["tail_id"], merged["type"],
                          _graph_edge_props(rel_props))

    return {"entities": len(ent_clusters), "relations": len(rel_clusters)}
