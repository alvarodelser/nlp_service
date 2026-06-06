from pipelines.relations.neo4j_io import Graph


class _FakeSession:
    def __init__(self, calls):
        self._calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, cypher, **params):
        self._calls.append((cypher, params))


class _FakeDriver:
    def __init__(self):
        self.calls = []

    def session(self):
        return _FakeSession(self.calls)


def test_upsert_node_merges_by_id_with_type_label():
    drv = _FakeDriver()
    g = Graph("financial_flows", driver=drv)
    g.upsert_node("f1", {"type": "PERSON", "canonical_name": "Ana", "description": "d"})
    cypher, params = drv.calls[0]
    assert "MERGE" in cypher
    assert "PERSON" in cypher                       # label applied
    assert params["id"] == "f1"
    assert params["props"]["canonical_name"] == "Ana"


def test_upsert_edge_matches_endpoints_and_merges_typed_edge():
    drv = _FakeDriver()
    g = Graph("financial_flows", driver=drv)
    g.upsert_edge("r1", "f1", "f2", "PAYMENT_TO", {"description": "d", "confidence": 0.9})
    cypher, params = drv.calls[0]
    assert "MATCH" in cypher and "MERGE" in cypher
    assert "PAYMENT_TO" in cypher
    assert params["head"] == "f1"
    assert params["tail"] == "f2"
    assert params["rid"] == "r1"
    assert params["props"]["confidence"] == 0.9


def test_label_backticks_are_stripped():
    drv = _FakeDriver()
    g = Graph("financial_flows", driver=drv)
    g.upsert_node("f1", {"type": "PER`SON", "canonical_name": "x"})
    cypher, _ = drv.calls[0]
    assert "SET n:`PERSON`" in cypher        # injected backtick stripped, label delimited safely
    assert "PER`SON" not in cypher
