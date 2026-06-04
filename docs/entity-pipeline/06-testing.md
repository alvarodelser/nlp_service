# Entity Pipeline — Testing Strategy

## Guiding principle

The pipeline is designed so that LLM calls are isolated behind thin client functions.
Everything else — schema validation, span arithmetic, edge merging, disambiguation scoring,
prompt building — is pure Python and fully unit-testable without any network access or
model. Tests are split accordingly.

```
Unit tests (no network, no model)
  nlp/schema.py          ← YAML loading, validation, schema emission, system prompt
  nlp/normalizer/merger.py  ← edge dedup, attribute merging, endpoint resolution
  nlp/dedup/disambiguator.py  ← scoring, alias bonus, three-tier decision
  nlp/entity_extractor/service.py (validation logic)  ← post-extraction checks

Integration tests (mock Ollama via httpx)
  nlp/entity_extractor/service.py  ← full run() with canned LLM response
  nlp/normalizer/service.py        ← clustering call with canned LLM response
  nlp/summarizer/service.py        ← describe_entity() with canned LLM response

Eval notebooks (human-reviewed, not in CI)
  eval/07_entity_extract_eval.ipynb
  eval/08_normalize_eval.ipynb
  eval/09_e2e_entity_pipeline.ipynb
```

---

## Mocking Ollama

All LLM calls go through `ollama_client.extract()` or `ollama_client.generate()`.
Mock at the `httpx.post` level so the client code itself is exercised:

```python
# tests/conftest.py addition
import json
import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_ollama_extract(request):
    """Fixture that returns a canned extraction payload.

    Usage:
        @pytest.mark.parametrize("mock_ollama_extract", [CANNED_PAYLOAD], indirect=True)
        def test_...(mock_ollama_extract): ...
    """
    payload = request.param
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "message": {"content": json.dumps(payload)}
    }
    with patch("httpx.post", return_value=response) as mock:
        yield mock
```

For summarizer tests the same fixture applies; the response shape differs
(`body["response"]` vs `body["message"]["content"]`) — check which client you're mocking.

---

## 1. Schema loader (`nlp/schema.py`)

No LLM, no network. Tests live in `tests/test_schema.py`.

### Fixtures

```
tests/fixtures/schemas/
  valid_financial_flows.yaml   ← minimal valid schema for testing
  missing_head_types.yaml      ← head_type references undeclared entity type
  bad_visual_weight.yaml       ← edge_weight names non-existent attribute
  no_version.yaml              ← missing required top-level key
```

### Key test cases

```python
def test_entity_type_names_includes_novel():
    s = load_schema("valid_financial_flows.yaml")
    assert "__NOVEL__" in s.entity_type_names()

def test_relation_type_names_includes_unclassified():
    s = load_schema("valid_financial_flows.yaml")
    assert "__UNCLASSIFIED__" in s.relation_type_names()

def test_required_entity_attrs():
    s = load_schema("valid_financial_flows.yaml")
    # financial_flows has no required entity attrs — all optional
    assert s.required_entity_attrs("PERSON") == []

def test_required_relation_attrs():
    s = load_schema("valid_financial_flows.yaml")
    assert set(s.required_relation_attrs("PAYMENT_TO")) == {"amount", "currency"}

def test_all_entity_attribute_keys_is_union():
    s = load_schema("valid_financial_flows.yaml")
    keys = s.all_entity_attribute_keys()
    assert "nationality" in keys       # PERSON
    assert "jurisdiction" in keys      # ORGANIZATION
    assert "account_type" in keys      # FINANCIAL_ENTITY

def test_extraction_schema_entity_has_attributes():
    s = load_schema("valid_financial_flows.yaml")
    schema = s.extraction_schema()
    entity_props = schema["properties"]["entities"]["items"]["properties"]
    assert "attributes" in entity_props
    assert "jurisdiction" in entity_props["attributes"]["properties"]

def test_system_prompt_contains_required_marker():
    s = load_schema("valid_financial_flows.yaml")
    prompt = s.system_prompt()
    assert "REQUIRED" in prompt
    assert "amount" in prompt
    assert "currency" in prompt

def test_validation_rejects_unknown_head_type():
    with pytest.raises(ValueError, match="head_type"):
        load_schema("missing_head_types.yaml")

def test_validation_rejects_bad_visual_weight():
    with pytest.raises(ValueError, match="edge_weight"):
        load_schema("bad_visual_weight.yaml")

def test_weaviate_spec_name_includes_version():
    s = load_schema("valid_financial_flows.yaml")
    spec = s.weaviate_collection_spec()
    assert "financial_flows" in spec["name"]
    assert "v1" in spec["name"]        # from schema version field
```

---

## 2. Edge merger (`nlp/normalizer/merger.py`)

Pure Python. Tests live in `tests/test_normalizer_merger.py`.

### Key test cases

```python
# helpers
def make_relation(head_id, rel_type, tail_id, **attrs):
    return ExtractedRelation(
        head_local_id=head_id, relation=rel_type, tail_local_id=tail_id,
        description="test", evidence_span={"start": 0, "end": 10},
        attributes=RelationAttributes(**attrs), confidence=0.9,
    )

def test_identical_triples_merge_to_one():
    edges = [make_relation("e1", "PAYMENT_TO", "e2", amount=100.0, currency="EUR"),
             make_relation("e1", "PAYMENT_TO", "e2", amount=100.0, currency="EUR")]
    assert len(merge_edges(edges)) == 1

def test_amounts_sum_on_dedup():
    edges = [make_relation("e1", "PAYMENT_TO", "e2", amount=100.0, currency="EUR"),
             make_relation("e1", "PAYMENT_TO", "e2", amount=50.0,  currency="EUR")]
    merged = merge_edges(edges)
    assert merged[0].attributes.amount == 150.0

def test_mixed_currency_flagged():
    edges = [make_relation("e1", "PAYMENT_TO", "e2", amount=100.0, currency="EUR"),
             make_relation("e1", "PAYMENT_TO", "e2", amount=50.0,  currency="USD")]
    merged = merge_edges(edges)
    assert merged[0].attributes.currency == "MIXED"

def test_evidence_spans_accumulated():
    edges = [make_relation("e1", "OWNS", "e2"),
             make_relation("e1", "OWNS", "e2")]
    merged = merge_edges(edges)
    assert len(merged[0].evidence) == 2

def test_distinct_triples_not_merged():
    edges = [make_relation("e1", "PAYMENT_TO", "e2", amount=100.0, currency="EUR"),
             make_relation("e1", "PAYMENT_TO", "e3", amount=50.0,  currency="EUR")]
    assert len(merge_edges(edges)) == 2

def test_endpoint_resolution_exact_match():
    # "Acme Holdings" in cluster → gets local_id "e1"
    clusters = [LocalEntity(local_id="e1", mentions=[EntityMention(text="Acme Holdings", ...)])]
    relations = [ExtractedRelation(head="Acme Holdings", tail="Musk", ...)]
    resolved = resolve_endpoints(relations, clusters)
    assert resolved[0].head_local_id == "e1"

def test_endpoint_resolution_normalised_match():
    # "acme holdings" (lowercased) matches cluster with "Acme Holdings"
    clusters = [LocalEntity(local_id="e1", mentions=[EntityMention(text="Acme Holdings", ...)])]
    relations = [ExtractedRelation(head="acme holdings", tail="Musk", ...)]
    resolved = resolve_endpoints(relations, clusters)
    assert resolved[0].head_local_id == "e1"

def test_unresolved_endpoint_flagged():
    clusters = []
    relations = [ExtractedRelation(head="Unknown Entity", tail="B", ...)]
    resolved = resolve_endpoints(relations, clusters)
    assert resolved[0].head_local_id is None
```

---

## 3. Disambiguator (`nlp/dedup/disambiguator.py`)

Pure logic — candidates are passed in, no Weaviate or LLM dependency in the core
`disambiguate()` function. Tests live in `tests/test_disambiguator.py`.

### Key test cases

```python
def make_candidate(score, name="Acme", aliases=None, subtype=None):
    return Candidate(canonical_id="c1", canonical_name=name, type="ORGANIZATION",
                     subtype=subtype, description="", aliases=aliases or [], score=score)

def make_entity(name="Acme Holdings", subtype=None):
    return LocalEntity(local_id="e1", canonical_name=name, type="ORGANIZATION",
                       subtype=subtype, description="", mentions=[])

def test_auto_merge_above_threshold():
    result = disambiguate(make_entity(), [1.0], [make_candidate(0.95)],
                          merge_threshold=0.92, review_low=0.75, llm_adjudicate=False)
    assert result.decision == "merge"
    assert result.canonical_id == "c1"

def test_auto_create_below_review_low():
    result = disambiguate(make_entity(), [1.0], [make_candidate(0.60)],
                          merge_threshold=0.92, review_low=0.75, llm_adjudicate=False)
    assert result.decision == "create"
    assert result.canonical_id is None

def test_auto_create_no_candidates():
    result = disambiguate(make_entity(), [1.0], [],
                          merge_threshold=0.92, review_low=0.75, llm_adjudicate=False)
    assert result.decision == "create"

def test_alias_bonus_can_reach_merge_threshold():
    # score=0.80 normally → review; alias match adds +0.15 → 0.95 → merge
    candidate = make_candidate(0.80, name="Acme Holdings", aliases=["Acme Holdings"])
    result = disambiguate(make_entity("Acme Holdings"), [1.0], [candidate],
                          merge_threshold=0.92, review_low=0.75, llm_adjudicate=False)
    assert result.decision == "merge"

def test_alias_bonus_capped_at_one():
    candidate = make_candidate(0.98, aliases=["Acme Holdings"])
    result = disambiguate(make_entity("Acme Holdings"), [1.0], [candidate],
                          merge_threshold=0.92, review_low=0.75, llm_adjudicate=False)
    assert result.confidence <= 1.0

def test_subtype_mismatch_penalty():
    candidate = make_candidate(0.93, subtype="BANK")
    entity    = make_entity(subtype="SHELL_COMPANY")
    result = disambiguate(entity, [1.0], [candidate],
                          merge_threshold=0.92, review_low=0.75, llm_adjudicate=False)
    # penalty drops 0.93 below threshold → not auto-merge
    assert result.decision in ("review", "create")

def test_llm_adjudicate_yes_merges(mock_ollama_extract):
    # mock LLM returns {"verdict": "yes"}
    candidate = make_candidate(0.83)
    result = disambiguate(make_entity(), [1.0], [candidate],
                          merge_threshold=0.92, review_low=0.75, llm_adjudicate=True)
    assert result.decision == "merge"

def test_llm_adjudicate_no_creates(mock_ollama_extract):
    # mock LLM returns {"verdict": "no"}
    ...

def test_llm_adjudicate_unsure_reviews(mock_ollama_extract):
    # mock LLM returns {"verdict": "unsure"}
    ...

def test_llm_error_returns_review(mock_ollama_raises):
    # Ollama unavailable → graceful degradation to "review"
    candidate = make_candidate(0.83)
    result = disambiguate(make_entity(), [1.0], [candidate],
                          merge_threshold=0.92, review_low=0.75, llm_adjudicate=True)
    assert result.decision == "review"
```

---

## 4. Entity extractor post-validation (`nlp/entity_extractor/service.py`)

The LLM call is mocked; the validation logic after it is what's under test.
Tests live in `tests/test_entity_extractor.py`.

### Key test cases

```python
CANNED_EXTRACTION = {
    "entities": [{
        "name": "María García", "type": "PERSON", "subtype": "POLITICIAN",
        "description": "Minister of Finance",
        "span": {"start": 5, "end": 17},
        "attributes": {"role_title": "Minister of Finance", "nationality": None, ...},
        "confidence": 0.95
    }],
    "relations": [{
        "head": "María García", "relation": "PAYMENT_TO", "tail": "Acme Holdings",
        "description": "wired €4.2M",
        "evidence_span": {"start": 5, "end": 80},
        "attributes": {"amount": 4200000.0, "currency": "EUR", "date": None, "direction": None, ...},
        "confidence": 0.88
    }]
}

@pytest.mark.parametrize("mock_ollama_extract", [CANNED_EXTRACTION], indirect=True)
def test_successful_extraction_returns_entities(mock_ollama_extract, schema):
    req = EntityExtractRequest(doc_id="d1", chunk_id="d1_000", text="...",
                               char_start=0, char_end=200, use_case="financial_flows")
    resp = service.run(req, schema)
    assert len(resp.entities) == 1
    assert resp.entities[0].name == "María García"

def test_span_beyond_text_is_clamped(mock_ollama_extract, schema):
    # entity span.end > len(text) → clamped, confidence unchanged
    payload = deep_copy(CANNED_EXTRACTION)
    payload["entities"][0]["span"]["end"] = 9999
    # inject payload, run, assert span clamped to len(text)

def test_inverted_span_zeroes_confidence(mock_ollama_extract, schema):
    payload = deep_copy(CANNED_EXTRACTION)
    payload["entities"][0]["span"] = {"start": 50, "end": 10}
    resp = service.run_with_payload(payload, schema, chunk_text="...")
    assert resp.entities[0].confidence == 0.0

def test_missing_required_relation_attr_penalises_confidence(schema):
    payload = deep_copy(CANNED_EXTRACTION)
    payload["relations"][0]["attributes"]["amount"]   = None
    payload["relations"][0]["attributes"]["currency"] = None
    resp = service.run_with_payload(payload, schema, chunk_text="...")
    assert resp.relations[0].confidence == pytest.approx(0.88 * 0.6)

def test_subtype_mismatch_nulled(schema):
    payload = deep_copy(CANNED_EXTRACTION)
    payload["entities"][0]["subtype"] = "BANK"   # BANK is not a PERSON subtype
    resp = service.run_with_payload(payload, schema, chunk_text="...")
    assert resp.entities[0].subtype is None

def test_ollama_unavailable_raises_503(client, mock_ollama_raises):
    response = client.post("/entity-extract", json={...})
    assert response.status_code == 503
```

---

## 5. Summarizer `describe_entity()`

Append to `tests/test_summarize.py` (existing file). Tests reuse the existing
`mock_ollama_extract` fixture pattern.

```python
CANNED_DESCRIPTION = {"description": "Acme Holdings is a Delaware-registered shell company..."}

@pytest.mark.parametrize("mock_ollama_extract", [CANNED_DESCRIPTION], indirect=True)
def test_describe_entity_returns_description(mock_ollama_extract):
    result = service.describe_entity(
        name="Acme Holdings", entity_type="ORGANIZATION", subtype="SHELL_COMPANY",
        evidence=["[doc_id: abc, 2024-03] Acme Holdings received €4.2M..."]
    )
    assert "description" in result
    assert len(result["description"]) > 0

def test_describe_entity_empty_evidence_raises():
    with pytest.raises(HTTPException):
        # tested via router with empty evidence list
        ...

def test_describe_entity_uses_entity_schema_not_article_schema(mock_ollama_extract):
    # verify the httpx.post call received the entity schema, not the article schema
    mock_ollama_extract.assert_called_once()
    call_body = json.loads(mock_ollama_extract.call_args[1]["json"]["format"])
    # entity schema has "description" key, NOT "headline"
    assert "description" in call_body["required"]
    assert "headline" not in call_body.get("required", [])
```

---

## Eval fixtures

Following the existing pattern in `eval/fixtures/`, add JSON fixture files for
human-reviewed ground-truth cases:

```
eval/fixtures/
  entity_extract_cases.json    ← [{chunk_text, use_case, expected_entities[], expected_relations[]}]
  normalize_cases.json         ← [{doc_id, chunks[], expected_clusters[], expected_edges[]}]
  disambiguate_cases.json      ← [{local_entity, candidates[], expected_decision}]
```

### `entity_extract_cases.json` entry format

```json
{
  "id": "fin_001",
  "chunk_text": "La ministra María García transfirió 4,2 millones de euros a Acme Holdings, una sociedad registrada en Delaware, en marzo de 2024.",
  "use_case": "financial_flows",
  "expected_entities": [
    {"name": "María García", "type": "PERSON", "subtype": "POLITICIAN"},
    {"name": "Acme Holdings", "type": "ORGANIZATION", "subtype": "SHELL_COMPANY"}
  ],
  "expected_relations": [
    {
      "head": "María García",
      "relation": "PAYMENT_TO",
      "tail": "Acme Holdings",
      "attributes": {"amount": 4200000, "currency": "EUR", "date": "2024-03"}
    }
  ]
}
```

Eval notebooks score precision/recall of entities and relations against these fixtures,
matching on `(name, type)` for entities and `(head, relation, tail)` for relations. Attribute
scoring uses a separate per-attribute exact-match check so you can see which attributes the
LLM struggles to fill.

---

## Running tests

```bash
# Unit + integration (mocked Ollama) — fast, runs in CI
pytest tests/test_schema.py tests/test_normalizer_merger.py tests/test_disambiguator.py tests/test_entity_extractor.py -v

# Full test suite including existing modules
pytest -v

# Eval notebooks (manual, not CI)
jupyter nbconvert --to notebook --execute eval/07_entity_extract_eval.ipynb
```

---

## What is NOT tested here

- **Ollama model quality** — whether the LLM actually extracts the right entities is
  evaluated in the eval notebooks with real Ollama calls, not in pytest.
- **Weaviate integration** — candidate retrieval is the orchestrator's concern. The
  disambiguator receives candidates as input and is tested with canned candidates.
- **Neo4j / persistence** — out of scope for this pipeline iteration.
