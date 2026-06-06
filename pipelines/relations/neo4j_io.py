"""Neo4j writes for Phase B only. Canonical entities -> nodes (label = type), canonical relations
-> typed edges, keyed by the final Weaviate id so the graph and vector store stay joinable. The
neo4j driver is imported lazily so the rest of the pipeline (and its tests) run without the package
or a live database; tests inject a driver."""
import os
import re

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

_IDENT_RE = re.compile(r"[^A-Za-z0-9_]")


def _ident(name: str, fallback: str) -> str:
    """Sanitize a label / relationship type for safe Cypher interpolation."""
    cleaned = _IDENT_RE.sub("", name or "")
    return cleaned or fallback


class Graph:
    def __init__(self, use_case: str, driver=None) -> None:
        self.use_case = use_case
        self._driver = driver or self._connect()

    @staticmethod
    def _connect():
        from neo4j import GraphDatabase            # lazy — only when a real connection is needed
        return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def upsert_node(self, cid: str, props: dict) -> None:
        label = _ident(props.get("type", ""), "Entity")
        cypher = f"MERGE (n {{id: $id}}) SET n += $props SET n:`{label}`"
        with self._driver.session() as s:
            s.run(cypher, id=cid, props=props)

    def upsert_edge(self, rid: str, head_id: str, tail_id: str, rel_type: str, props: dict) -> None:
        rtype = _ident(rel_type, "RELATED_TO")
        cypher = (f"MATCH (h {{id: $head}}), (t {{id: $tail}}) "
                  f"MERGE (h)-[r:`{rtype}` {{id: $rid}}]->(t) SET r += $props")
        with self._driver.session() as s:
            s.run(cypher, head=head_id, tail=tail_id, rid=rid, props=props)
