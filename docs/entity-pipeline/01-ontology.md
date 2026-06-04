# Extraction Schema (Type Catalogue) — Design & Implementation

## Purpose

Single source of truth for a use-case's entity types, relation types, their extractable
attributes, and visual encoding hints. Every module that touches the extraction pipeline
imports the schema rather than hardcoding types. The loader validates the YAML at startup,
then emits three derived artefacts on demand — all by **mechanical rule, no intelligence
required**:

| Artefact | Consumer | How it is derived |
|---|---|---|
| **JSON Schema** (`format=`) | Ollama constrained decoding in Pass 1 | Entity type names → enum; all attribute keys across all types → nullable properties union |
| **System prompt section** | LLM extraction prompt | Type descriptions + per-type attribute requirements table, templated from YAML fields |
| **Weaviate collection spec** | Persistence on first run / migration | Entity type names + attribute keys → Weaviate property definitions |

One process may load multiple schemas (one per active use case).

### Phase 0 — closed, predetermined schema

In Phase 0 the schema is fixed before any documents are processed. The entity and relation
type sets are decided by the analyst, written to YAML, and treated as a closed world: the
LLM extracts only from those types, and there are no escape hatches (`__NOVEL__`,
`__UNCLASSIFIED__` are not used).

### Future phases

- **Phase 1 — schema editor UI**: the YAML-backed schema is exposed in a UI where the
  analyst can add or edit types, attributes, and visual hints without touching files. Changes
  version-bump the schema and trigger a Weaviate collection migration.

- **Phase 2 — semi-supervised schema generation**: an agent reads a sample of the corpus,
  proposes a candidate schema (entity types, relations, attributes it observes), and presents
  it for analyst review and approval. Once approved the schema is locked and passed to Phase 0
  processing. The agent proposes; the human decides.

---

## File layout

```
config/
  ontologies/
    financial_flows.yaml      ← first use case (journalism / payments)
    <next_use_case>.yaml

nlp/
  schema.py                   ← loader, validator, schema emitters
                                 (referred to as "ontology" in older docs — same file)
```

No router or endpoint. Other modules call `nlp.schema.load(use_case)`.

---

## Ontology YAML format

```yaml
# config/ontologies/financial_flows.yaml
version: "1"
use_case: financial_flows

entity_types:
  - name: PERSON
    description: >
      A natural person referenced by name, title, pronoun, or alias.
      Extract even when only a role is mentioned ("the CEO") if the
      surrounding context makes the individual identifiable.
    visual:
      color:   "#F5A623"
      shape:   circle
      size_by: mention_count     # derived at render time, not extracted
    subtypes:
      - name: POLITICIAN
        description: Elected or appointed public official.
        visual: {color: "#E74C3C"}
      - name: EXECUTIVE
        description: Senior corporate officer (CEO, CFO, board member).
        visual: {color: "#F5A623"}
      - name: INTERMEDIARY
        description: Lawyer, accountant, nominee director, or agent acting
                     on behalf of another party.
        visual: {color: "#9B59B6", border: dashed}
    attributes:
      nationality:
        type:     string
        required: false
        description: Country of citizenship or nationality (ISO 3166-1 alpha-2 if known).
      role_title:
        type:     string
        required: false
        description: Most specific role or title stated in the text (e.g. "Minister of Finance").
      date_of_birth:
        type:     string
        required: false
        description: ISO 8601 date or partial (e.g. "1965" or "1965-03").

  - name: ORGANIZATION
    description: >
      Any formal or informal group acting as a unit: company, fund,
      foundation, government body, or criminal network.
    visual:
      color:   "#2E86AB"
      shape:   square
      size_by: mention_count
    subtypes:
      - name: SHELL_COMPANY
        description: Legal entity with no active business operations,
                     typically used to hold assets or obscure ownership.
        visual: {color: "#E67E22", border: dashed}
      - name: BANK
        description: Licensed deposit-taking institution.
        visual: {color: "#27AE60"}
      - name: FUND
        description: Investment vehicle (hedge fund, PE fund, trust).
        visual: {color: "#2980B9"}
      - name: GOVERNMENT_BODY
        description: Regulatory authority, ministry, or state-owned enterprise.
        visual: {color: "#8E44AD"}
    attributes:
      jurisdiction:
        type:     string
        required: false
        description: Country or territory of incorporation (ISO 3166-1 alpha-2 if known).
        visual:   node_label_secondary
      registration_number:
        type:     string
        required: false
        description: Company registration or tax ID number if stated.
      founding_date:
        type:     string
        required: false
        description: ISO 8601 date or partial year.

  - name: FINANCIAL_ENTITY
    description: >
      An account, instrument, or financial product: bank account, wire
      transfer, bond, loan, equity stake.
    visual:
      color: "#16A085"
      shape: diamond
    subtypes: []
    attributes:
      account_type:
        type:     string
        required: false
        description: "Type of financial instrument: bank_account, wire, bond, loan, equity, crypto_wallet, other."
      institution:
        type:     string
        required: false
        description: Name of the bank or institution holding or issuing this entity.
      currency:
        type:     string
        required: false
        description: ISO 4217 currency code if stated.

  - name: LOCATION
    description: >
      A jurisdiction, country, city, address, or offshore zone relevant
      to where entities are registered or transactions occur.
    visual:
      color: "#7F8C8D"
      shape: hexagon
    subtypes: []
    attributes:
      country_code:
        type:     string
        required: false
        description: ISO 3166-1 alpha-2 country code.
      jurisdiction_type:
        type:     string
        required: false
        description: "Classification: offshore_haven, eu_member, sanctioned, other."

  - name: EVENT
    description: >
      A dateable occurrence: a meeting, a transaction, a filing, a
      court ruling.
    visual:
      color: "#BDC3C7"
      shape: triangle
    subtypes: []
    attributes:
      date:
        type:     string
        required: false
        description: ISO 8601 date or partial.
        visual:   timeline
      event_type:
        type:     string
        required: false
        description: "Classification: meeting, transaction, filing, ruling, other."

relation_types:
  - name: PAYMENT_TO
    description: >
      One party transfers money or assets to another. Capture
      amount, currency, date, and mechanism (wire, cash, crypto) in
      attributes.
    head_types: [PERSON, ORGANIZATION, FINANCIAL_ENTITY]
    tail_types: [PERSON, ORGANIZATION, FINANCIAL_ENTITY]
    visual:
      color:        "#E74C3C"
      direction:    left-to-right
      edge_weight:  amount
      edge_label:   "{amount} {currency}"
    attributes:
      amount:
        type:     number
        required: true
        visual:   edge_weight
        description: Numeric amount, no currency symbol.
      currency:
        type:     string
        required: true
        description: ISO 4217 code (USD, EUR…).
      date:
        type:     string
        required: false
        visual:   timeline
        description: ISO 8601 date or partial (2024-Q1).
      direction:
        type:     string
        required: false
        description: "incoming | outgoing | unknown"

  - name: OWNS
    description: One party holds an ownership stake in another.
    head_types: [PERSON, ORGANIZATION]
    tail_types: [ORGANIZATION, FINANCIAL_ENTITY]
    visual:
      color:       "#2E86AB"
      direction:   top-to-bottom
      edge_weight: stake_pct
      edge_label:  "{stake_pct}%"
    attributes:
      stake_pct:
        type:     number
        required: false
        visual:   edge_weight
        description: Ownership percentage if stated.
      date:
        type:     string
        required: false
        visual:   timeline

  - name: CONTROLS
    description: >
      De-facto control without necessarily owning a majority stake.
    head_types: [PERSON, ORGANIZATION]
    tail_types: [ORGANIZATION]
    visual:
      color:     "#8E44AD"
      direction: top-to-bottom
    attributes: {}

  - name: MEMBER_OF
    description: A person holds a formal role inside an organisation.
    head_types: [PERSON]
    tail_types: [ORGANIZATION]
    visual:
      color: "#27AE60"
    attributes:
      role:
        type:     string
        required: false
        description: Job title or board position.
      date_from:
        type:     string
        required: false
        visual:   timeline
      date_to:
        type:     string
        required: false
        visual:   timeline

  - name: REGISTERED_IN
    description: An entity is legally incorporated or registered in a jurisdiction.
    head_types: [ORGANIZATION, FINANCIAL_ENTITY]
    tail_types: [LOCATION]
    visual:
      color: "#7F8C8D"
    attributes:
      date:
        type:     string
        required: false
        visual:   timeline

  - name: PARTY_TO
    description: A person or organisation is a named party to an event (deal, lawsuit, filing).
    head_types: [PERSON, ORGANIZATION]
    tail_types: [EVENT]
    visual:
      color: "#F39C12"
    attributes:
      role:
        type:     string
        required: false
        description: "plaintiff | defendant | signatory | witness | other"

```

### Validation rules (enforced at load time)

| Rule | Check |
|---|---|
| `name` is SCREAMING_SNAKE_CASE | regex |
| All `head_types` / `tail_types` values are declared entity type names | set membership |
| Each `attributes` entry has `type` in `{string, number, boolean}` | enum |
| Each `attributes` entry has `required` as a boolean | type check |
| `visual.edge_weight` names an attribute defined on the same type | key lookup |
| `version` is a string that can be compared with `packaging.version` | cast |
| No duplicate names within `entity_types` or `relation_types` | set |

---

## Python API

```python
from nlp.schema import load, ExtractionSchema

schema: ExtractionSchema = load("financial_flows")  # cached after first call

schema.entity_type_names()                     # -> ["PERSON", "ORGANIZATION", ...]
schema.subtype_names()                         # -> ["POLITICIAN", "EXECUTIVE", ...] (pooled)
schema.relation_type_names()                   # -> ["PAYMENT_TO", "OWNS", ...]
schema.entity_type("ORGANIZATION")             # -> EntityType(...)
schema.relation_type("PAYMENT_TO")             # -> RelationType(...)
schema.subtypes_for("ORGANIZATION")            # -> ["SHELL_COMPANY", "BANK", ...]
schema.validate_subtype("POLITICIAN", parent="PERSON")  # -> True

# Required attribute keys for post-extraction validation
schema.required_entity_attrs("ORGANIZATION")   # -> [] (none required in financial_flows)
schema.required_relation_attrs("PAYMENT_TO")   # -> ["amount", "currency"]

# Three rule-based emitters — no intelligence, just loops over the parsed YAML
schema.extraction_schema()             # -> dict  (JSON Schema for Ollama format=)
schema.system_prompt()                 # -> str   (per-type descriptions + attribute requirements)
schema.weaviate_collection_spec()      # -> dict  (Weaviate collection definition)
```

### `extraction_schema()` output structure

The grammar enforces the outer shape of the output — required fields, type/relation enums,
span structure. The `attributes` field is left as a free JSON object: the grammar only
guarantees it is a valid object, nothing more. Attribute keys and value types are guided
entirely by `system_prompt()` and validated in application code after the fact.

This avoids polluting every entity and relation with null keys that belong to other types.

```jsonc
{
  "type": "object",
  "required": ["entities", "relations"],
  "additionalProperties": false,
  "properties": {
    "entities": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["name", "type", "description", "span", "attributes", "confidence"],
        "additionalProperties": false,
        "properties": {
          "name":        {"type": "string"},
          "type":        {"enum": ["PERSON", "ORGANIZATION", ...]},
          "subtype":     {"oneOf": [{"enum": ["POLITICIAN", ...]}, {"type": "null"}]},
          "description": {"type": "string"},
          "span":        {"type": "object", "required": ["start","end"],
                          "properties": {"start": {"type":"integer"}, "end": {"type":"integer"}}},
          "attributes":  {"type": "object"},   // free — keys vary per entity type
          "confidence":  {"type": "number", "minimum": 0, "maximum": 1}
        }
      }
    },
    "relations": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["head", "relation", "tail", "description", "evidence_span", "attributes", "confidence"],
        "additionalProperties": false,
        "properties": {
          "head":          {"type": "string"},
          "relation":      {"enum": ["PAYMENT_TO", "OWNS", ...]},
          "tail":          {"type": "string"},
          "description":   {"type": "string"},
          "evidence_span": {"type": "object", "required": ["start","end"],
                            "properties": {"start": {"type":"integer"}, "end": {"type":"integer"}}},
          "attributes":    {"type": "object"},   // free — keys vary per relation type
          "confidence":    {"type": "number", "minimum": 0, "maximum": 1}
        }
      }
    }
  }
}
```

### `system_prompt()` output — per-type attribute section

Each type only sees its own attributes. The LLM never sees attribute names from other types
while writing a given entity or relation, so there is no cross-contamination.

```
ENTITY TYPES
============
PERSON: A natural person referenced by name, title, pronoun, or alias...
  POLITICIAN: Elected or appointed public official.
  EXECUTIVE: Senior corporate officer (CEO, CFO, board member).
  INTERMEDIARY: Lawyer, accountant, nominee director...
ORGANIZATION: Any formal or informal group...
  ...

RELATION TYPES
==============
PAYMENT_TO  (PERSON | ORGANIZATION | FINANCIAL_ENTITY) → (PERSON | ORGANIZATION | FINANCIAL_ENTITY)
  One party transfers money or assets to another.
OWNS  (PERSON | ORGANIZATION) → (ORGANIZATION | FINANCIAL_ENTITY)
  One party holds an ownership stake in another.
...

ATTRIBUTE REQUIREMENTS
======================
  PERSON               → optional: role_title, nationality, date_of_birth
  ORGANIZATION         → optional: jurisdiction, registration_number, founding_date
  FINANCIAL_ENTITY     → optional: account_type, institution, currency
  LOCATION             → optional: country_code, jurisdiction_type
  EVENT                → optional: date, event_type

  PAYMENT_TO           → REQUIRED: amount, currency;  optional: date, direction
  OWNS                 → optional: stake_pct, date
  MEMBER_OF            → optional: role, date_from, date_to
  REGISTERED_IN        → optional: date
  PARTY_TO             → optional: role
```

Post-extraction, application code calls `schema.required_relation_attrs("PAYMENT_TO")`
to check which attributes must be non-null, and `schema.entity_type("PERSON").attributes`
to know what keys to expect and coerce to the right Python types.

### `weaviate_collection_spec()` output structure

```jsonc
{
  "name": "Entity_financial_flows_v1",
  "description": "Canonical entities for use case financial_flows, schema v1",
  "vectorizer_config": [{"vectorizer": {"none": {}}}],
  "properties": [
    {"name": "canonical_name",    "dataType": ["text"]},
    {"name": "type",              "dataType": ["text"]},
    {"name": "subtype",           "dataType": ["text"]},
    {"name": "description",       "dataType": ["text"]},
    {"name": "aliases",           "dataType": ["text[]"]},
    {"name": "doc_ids",           "dataType": ["text[]"]},
    {"name": "evidence_windows",  "dataType": ["text[]"]},   // text windows for description refresh
    {"name": "schema_version",    "dataType": ["text"]}
  ],
  "vectorIndexConfig": {"distance": "cosine"}
}
```

`evidence_windows` stores the pre-formatted text windows accumulated from every document
mention of this entity. Each item is a string of the form:

```
[doc_id: abc123, 2024-03-15] "Sentence before mention. The entity mention sentence. Sentence after."
```

The persistence module appends a new window on every merge or create. The caller reads
this array from Weaviate and passes it as `evidence` to `POST /summarize` with
`type: "entity"`. The NLP service itself never writes to or reads from Weaviate — it
only receives the evidence as input and returns a description.

A separate `Edge_{use_case}_v1` collection mirrors this structure for canonical relation
edges, carrying `head_canonical_id`, `relation`, `tail_canonical_id`, `description`,
`evidence_windows`, and `doc_ids`.

The collection name embeds the use case and schema version so incompatible schemas never
share a collection. Migrations create a new collection, backfill, then reroute writes.

---

## Implementation notes

### Caching

```python
_cache: dict[str, ExtractionSchema] = {}

def load(use_case: str) -> ExtractionSchema:
    if use_case not in _cache:
        path = _SCHEMA_DIR / f"{use_case}.yaml"
        _cache[use_case] = _load_and_validate(path)
    return _cache[use_case]
```

`_ONTOLOGY_DIR` defaults to `config/ontologies/`, overridable via `ONTOLOGY_DIR` env var.

### Post-extraction subtype validation

After the extractor returns, validate that each entity's `subtype` belongs to its parent
`type`. Entities with mismatched subtype get `subtype=null` and a `WARNING` log — do not
reject the whole extraction.

---

## Configuration

| Env var | Default | Description |
|---|---|---|
| `ONTOLOGY_DIR` | `config/ontologies` | Directory scanned for YAML files |

---

## Testing

See [06-testing.md](06-testing.md) §1 — pure unit tests, no mocking needed.
Test fixtures: `tests/fixtures/schemas/`.

---

## Dependencies

- `pyyaml` (already in requirements.txt)
- No new model or network call
