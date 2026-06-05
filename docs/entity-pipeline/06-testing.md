# Entity Pipeline — Testing Strategy

## Guiding principle

The pipeline is designed so that LLM calls are isolated behind thin client functions.
Everything else — schema validation, span arithmetic, edge merging, disambiguation scoring,
prompt building — is pure Python and fully unit-testable without any network access or
model. Tests are split accordingly.

```
Unit tests (no network, no model)
  nlp/schema.py          ← YAML loading, validation, schema emission, system prompt, Weaviate spec
  nlp/resolver/premerge.py + edges.py  ← pre-merge, relation rewrite by name, overlap collapse
  nlp/dedup/disambiguator.py  ← vector + subtype scoring, three-tier decision
  nlp/entity_extractor/service.py (validation logic)  ← post-extraction checks

Integration tests (mock Ollama via httpx)
  nlp/entity_extractor/service.py  ← full run() with canned LLM response
  nlp/resolver/service.py          ← refine call with canned LLM response
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
def test_entity_types_are_closed_world():
    s = load_schema("valid_financial_flows.yaml")
    names = s.entity_type_names()
    assert "PERSON" in names and "ORGANIZATION" in names
    assert "__NOVEL__" not in names          # closed-world: no escape hatch

def test_relation_types_are_closed_world():
    s = load_schema("valid_financial_flows.yaml")
    assert "__UNCLASSIFIED__" not in s.relation_type_names()

def test_all_entity_attribute_keys_is_union():
    s = load_schema("valid_financial_flows.yaml")
    keys = s.all_entity_attribute_keys()
    assert "nationality" in keys       # PERSON
    assert "jurisdiction" in keys      # ORGANIZATION
    assert "account_type" in keys      # FINANCIAL_ENTITY

def test_to_extraction_schema_projection():
    # 01 projects the rich YAML down to the slim ExtractionSchema the extractor consumes
    s = load_schema("valid_financial_flows.yaml")
    es = s.to_extraction_schema()
    person = next(t for t in es.entity_types if t.name == "PERSON")
    assert {st.name for st in person.subtypes} == {"POLITICIAN", "EXECUTIVE", "INTERMEDIARY"}
    assert {a.name for a in person.attributes} == {"nationality", "role_title", "date_of_birth"}
    assert {a.datatype for a in person.attributes} == {"string"}
    payment = next(t for t in es.relation_types if t.name == "PAYMENT_TO")
    assert "PERSON" in payment.head_types and "ORGANIZATION" in payment.tail_types
    # visual / required / version are dropped in the projection — not the extractor's concern
    assert not hasattr(person, "visual")

def test_weaviate_spec_includes_attribute_columns():
    # every declared entity attribute gets a property so it has somewhere to land at upsert
    s = load_schema("valid_financial_flows.yaml")
    props = {p["name"]: p["dataType"] for p in s.weaviate_collection_spec()["properties"]}
    assert {"canonical_name", "type", "description", "aliases", "doc_ids"} <= props.keys()
    assert "jurisdiction" in props and "nationality" in props   # from the schema attributes
    assert props["jurisdiction"] == ["text"]                    # string → text

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

## 2. Resolver (`nlp/resolver/`)

`premerge.py` and `edges.py` are pure Python (no LLM); the LLM refine is mocked. Tests live in
`tests/test_resolver.py`.

### Key test cases

```python
# --- pre-merge (premerge.py): exact-name grouping ---

def test_premerge_collapses_exact_name_repeats():
    ents = [EntityIn(name="Laura Méndez", type="PERSON", evidence="..."),
            EntityIn(name="Laura Méndez", type="PERSON", evidence="..."),
            EntityIn(name="Méndez",       type="PERSON", evidence="...")]
    cands = premerge(ents)
    assert len(cands) == 2          # the two "Laura Méndez" collapse; "Méndez" awaits refine

def test_premerge_never_merges_across_types():
    ents = [EntityIn(name="Delta", type="ORGANIZATION", evidence="..."),
            EntityIn(name="Delta", type="PERSON", evidence="...")]
    assert len(premerge(ents)) == 2

# --- relation rewrite + overlap collapse (edges.py), given a name→canonical map ---

NAME2CANON = {"Méndez": "Laura Méndez", "Laura Méndez": "Laura Méndez", "Delta": "Delta S.A."}

def make_rel(head, tail, rel="PAYMENT_TO", evidence="pagó 100 EUR", conf=0.9, **attrs):
    return RelationIn(head=head, tail=tail, type=rel, evidence=evidence,
                      attributes=attrs, confidence=conf)

def test_rewrite_maps_endpoints_to_canonical():
    out = resolve_relations([make_rel("Méndez", "Delta")], NAME2CANON)
    assert (out[0].head, out[0].tail) == ("Laura Méndez", "Delta S.A.")

def test_dangling_endpoint_dropped():
    assert resolve_relations([make_rel("Desconocido", "Delta")], NAME2CANON) == []

def test_overlapping_duplicates_collapse_to_one():
    # same canonical endpoints + same evidence (two overlapping chunks) → one relation
    rels = [make_rel("Méndez", "Delta", evidence="pagó 100 EUR"),
            make_rel("Laura Méndez", "Delta", evidence="pagó 100 EUR")]
    out = resolve_relations(rels, NAME2CANON)
    assert len(out) == 1 and len(out[0].evidence) == 2

def test_distinct_instances_not_merged():
    rels = [make_rel("Méndez", "Delta", evidence="pagó 100 EUR en marzo"),
            make_rel("Méndez", "Delta", evidence="pagó 50 EUR en mayo")]
    assert len(resolve_relations(rels, NAME2CANON)) == 2

def test_attributes_pass_through_and_max_confidence():
    rels = [make_rel("Méndez", "Delta", evidence="es dueño", conf=0.7, stake_pct=10),
            make_rel("Méndez", "Delta", evidence="es dueño", conf=0.9, stake_pct=12)]
    out = resolve_relations(rels, NAME2CANON)[0]
    assert out.attributes["stake_pct"] == 12   # higher-confidence instance; never aggregated
    assert out.confidence == 0.9               # max of collapsed duplicates
```

---

## 3. Disambiguator (`nlp/dedup/disambiguator.py`)

Pure logic — candidates are passed in, no Weaviate or LLM dependency in the core
`disambiguate()` function. Tests live in `tests/test_disambiguator.py`.

### Key test cases

```python
def make_candidate(score, name="Acme", subtype=None):
    return Candidate(canonical_id="c1", canonical_name=name, type="ORGANIZATION",
                     subtype=subtype, description="", score=score)

def make_entity(name="Delta S.A.", subtype=None):
    return ResolvedEntity(canonical_name=name, names=[name], type="ORGANIZATION",
                          subtype=subtype, evidence=[])

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

## 4. Entity extractor (`nlp/entity_extractor/`)

Stateless: `run(text, schema)`. Two pure-Python concerns (compile the schema; check the one
relational rule) plus one mocked-Ollama run. Tests live in `tests/test_entity_extractor.py`.

### 4a. Schema compiler (pure, no LLM)

```python
def test_compiler_builds_discriminated_union(extraction_schema):
    grammar, prompt = compile_schema(extraction_schema)   # ExtractionSchema → (grammar, prompt)
    # reasoning scratchpad is the first property (no-CoT workaround)
    assert grammar["properties"]["reasoning"]["type"] == "string"
    # one entity branch per type, each scoping ONLY its own attributes
    branches = grammar["properties"]["entities"]["items"]["oneOf"]
    by_type = {b["properties"]["type"]["const"]: b for b in branches}
    person_attrs = by_type["PERSON"]["properties"]["attributes"]["properties"]
    assert "nationality" in person_attrs         # PERSON owns this
    assert "jurisdiction" not in person_attrs     # ...and is NOT shown ORGANIZATION's attrs
    # relation endpoints are integer indices; the type catalogue is injected into the prompt
    pay = next(b for b in grammar["properties"]["relations"]["items"]["oneOf"]
               if b["properties"]["type"]["const"] == "PAYMENT_TO")
    assert pay["properties"]["head"]["type"] == "integer"
    assert "PERSON" in prompt
```

(subtype/attribute/datatype validity is *not* tested post-hoc — the grammar guarantees it at
generation, so there is nothing to check.)

### 4b. The one runtime rule: relation subject/object types

```python
SCHEMA = ...  # ExtractionSchema; PAYMENT_TO.head_types = PAYMENT_TO.tail_types = [PERSON, ORGANIZATION]

CANNED = {
    "reasoning": "Una ministra y una sociedad, con un pago entre ambas.",
    "entities": [
        {"name": "Laura Méndez", "mention_text": "Laura Méndez", "type": "PERSON",
         "subtype": "POLITICIAN",
         "evidence_text": "La ministra Laura Méndez transfirió 4,2 millones de euros a Delta S.A.",
         "attributes": {"role_title": "Ministra de Finanzas"}, "confidence": 0.95},
        {"name": "Delta S.A.", "mention_text": "Delta S.A.", "type": "ORGANIZATION",
         "subtype": None,
         "evidence_text": "La ministra Laura Méndez transfirió 4,2 millones de euros a Delta S.A.",
         "attributes": {}, "confidence": 0.9},
    ],
    "relations": [
        {"head": 0, "type": "PAYMENT_TO", "tail": 1, "subtype": None,
         "evidence_text": "Laura Méndez transfirió 4,2 millones de euros a Delta S.A.",
         "attributes": {"amount": 4200000.0, "currency": "EUR"}, "confidence": 0.88},
    ],
}

@pytest.mark.parametrize("mock_ollama", [CANNED], indirect=True)
def test_valid_extraction_passes_through(mock_ollama):
    resp = service.run(text="...", schema=SCHEMA)
    assert len(resp.entities) == 2
    assert resp.relations[0].type == "PAYMENT_TO"
    assert (resp.relations[0].head, resp.relations[0].tail) == (0, 1)

def test_out_of_range_endpoint_dropped():
    payload = deep_copy(CANNED); payload["relations"][0]["head"] = 7   # only 0 and 1 exist
    resp = service.run_with_payload(payload, SCHEMA)
    assert resp.relations == []

def test_type_incompatible_relation_dropped():
    # object becomes a LOCATION, but PAYMENT_TO.tail_types = [PERSON, ORGANIZATION]
    payload = deep_copy(CANNED)
    payload["entities"][1]["type"] = "LOCATION"; payload["entities"][1]["subtype"] = None
    resp = service.run_with_payload(payload, SCHEMA)
    assert resp.relations == []

def test_entities_are_never_dropped():
    payload = deep_copy(CANNED); payload["relations"][0]["head"] = 7
    resp = service.run_with_payload(payload, SCHEMA)
    assert len(resp.entities) == 2     # only the bad relation is dropped

def test_ollama_unavailable_raises_503(client, mock_ollama_raises):
    response = client.post("/entity-extract", json={"text": "...", "schema": {...}})
    assert response.status_code == 503
```

---

## 5. Summarizer `describe_entity()`

Append to `tests/test_summarize.py` (existing file). Tests reuse the existing
`mock_ollama_extract` fixture pattern.

```python
CANNED_DESCRIPTION = {"description": "Delta S.A. es una sociedad pantalla registrada en Delaware..."}

@pytest.mark.parametrize("mock_ollama_extract", [CANNED_DESCRIPTION], indirect=True)
def test_describe_entity_returns_description(mock_ollama_extract):
    result = service.describe_entity(
        name="Delta S.A.", entity_type="ORGANIZATION", subtype="SHELL_COMPANY",
        evidence=["[doc_id: abc, 2024-03] Delta S.A. recibió 4,2 M€ de la ministra..."]
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

Gold relations are authored by entity **name** for readability. Because the model now emits
`head`/`tail` as integer indices into its `entities` array, the eval harness first resolves
each predicted index back to that entity's `name`, then scores. Eval notebooks score
precision/recall of entities and relations against these fixtures, matching on `(name, type)`
for entities and `(head_name, relation, tail_name)` for relations. Attribute scoring uses a
separate per-attribute exact-match check so you can see which attributes the LLM struggles to
fill.

---

## Running tests

```bash
# Unit + integration (mocked Ollama) — fast, runs in CI
pytest tests/test_schema.py tests/test_resolver.py tests/test_disambiguator.py tests/test_entity_extractor.py -v

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
