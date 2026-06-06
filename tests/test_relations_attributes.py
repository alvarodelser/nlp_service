from pipelines.relations.attributes import build_attribute_index, join_entity_attributes


def _idx(records):
    # records: list of {name, attributes, confidence}
    return build_attribute_index(records)


def test_highest_confidence_non_null_wins():
    idx = _idx([
        {"name": "Banco X", "confidence": 0.9, "attributes": {"jurisdiction": "ES"}},
        {"name": "B. X", "confidence": 0.7, "attributes": {"jurisdiction": "FR"}},
    ])
    out = join_entity_attributes(["Banco X", "B. X"], idx, ["jurisdiction"])
    assert out == {"jurisdiction": "ES"}


def test_unions_across_attributes_from_different_records():
    idx = _idx([
        {"name": "X", "confidence": 0.9, "attributes": {"a": 1}},
        {"name": "X", "confidence": 0.8, "attributes": {"b": 2}},
    ])
    assert join_entity_attributes(["X"], idx, ["a", "b"]) == {"a": 1, "b": 2}


def test_skips_null_values_and_falls_through_to_lower_confidence():
    idx = _idx([
        {"name": "X", "confidence": 0.9, "attributes": {"a": None}},
        {"name": "X", "confidence": 0.5, "attributes": {"a": 5}},
    ])
    assert join_entity_attributes(["X"], idx, ["a"]) == {"a": 5}


def test_only_declared_attributes_are_kept():
    idx = _idx([{"name": "X", "confidence": 0.9, "attributes": {"a": 1, "undeclared": 9}}])
    assert join_entity_attributes(["X"], idx, ["a"]) == {"a": 1}


def test_unknown_names_yield_empty():
    idx = _idx([{"name": "X", "confidence": 0.9, "attributes": {"a": 1}}])
    assert join_entity_attributes(["Y"], idx, ["a"]) == {}


def test_collects_across_all_names_of_a_resolved_entity():
    idx = _idx([
        {"name": "Banco X", "confidence": 0.6, "attributes": {"jurisdiction": "ES"}},
        {"name": "BX S.A.", "confidence": 0.95, "attributes": {"jurisdiction": "FR"}},
    ])
    out = join_entity_attributes(["Banco X", "BX S.A."], idx, ["jurisdiction"])
    assert out == {"jurisdiction": "FR"}     # 0.95 across names wins
