# Extraction Schema (Type Catalogue) — Design & Implementation

## Purpose

Single source of truth for a use-case's entity types, relation types, their extractable
attributes, and visual encoding hints. Every module that touches the extraction pipeline
imports the schema rather than hardcoding types. The loader validates the YAML at startup,
then emits derived artefacts on demand:

- A **JSON Schema** consumed by Ollama's constrained decoding (`format=`) in Pass 1
- A **system prompt section** listing per-type attribute requirements for the LLM
- A **Weaviate collection spec** consumed by persistence when creating or migrating a collection

One process may load multiple schemas (one per active use case).

The schema is **open-world**: the LLM is guided toward known types but is never forced to
misclassify. Unknown entity types are extracted as `__NOVEL__`; unknown relation types as
`__UNCLASSIFIED__`. Both accumulate in a review queue and become candidates for schema
expansion.

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

  - name: __NOVEL__
    description: >
      Entity that does not fit any defined type. Use only when genuinely
      no other type applies. The schema team will review and potentially
      promote to a named type.
    visual: {color: "#BDC3C7", border: dashed}
    subtypes: []
    attributes: {}

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

  - name: __UNCLASSIFIED__
    description: >
      Relation that does not fit any defined type. Preserve the surface
      phrase in `description`. The schema team will review and potentially
      promote to a named type.
    head_types: []   # unconstrained
    tail_types: []
    visual: {color: "#BDC3C7", line: dashed}
    attributes: {}
```

### Validation rules (enforced at load time)

| Rule | Check |
|---|---|
| `name` is SCREAMING_SNAKE_CASE (or `__NOVEL__` / `__UNCLASSIFIED__`) | regex |
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

schema.entity_type_names()              # -> ["PERSON", "ORGANIZATION", ..., "__NOVEL__"]
schema.subtype_names()                  # -> ["POLITICIAN", "EXECUTIVE", ...] (all, pooled)
schema.relation_type_names()            # -> ["PAYMENT_TO", "OWNS", ..., "__UNCLASSIFIED__"]
schema.entity_type("ORGANIZATION")      # -> EntityType(...)
schema.relation_type("PAYMENT_TO")      # -> RelationType(...)
schema.validate_subtype("POLITICIAN", parent="PERSON")  # -> True

# All entity attribute keys across all types (union, for JSON Schema generation)
schema.all_entity_attribute_keys()      # -> ["nationality", "role_title", "jurisdiction", ...]

# All relation attribute keys across all types (union)
schema.all_relation_attribute_keys()    # -> ["amount", "currency", "date", "stake_pct", ...]

# Required attribute keys for a given type
schema.required_entity_attrs("PERSON")         # -> ["nationality", ...] (those with required: true)
schema.required_relation_attrs("PAYMENT_TO")   # -> ["amount", "currency"]

schema.extraction_schema()    # -> dict  (JSON Schema for Ollama format=, includes entity attributes)
schema.system_prompt()        # -> str   (type descriptions + per-type attribute requirements table)
schema.weaviate_collection_spec()  # -> dict  (Weaviate collection definition)
```

### `extraction_schema()` output structure

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
        "required": ["name", "type", "description", "span", "confidence"],
        "additionalProperties": false,
        "properties": {
          "name":        {"type": "string"},
          "type":        {"enum": ["PERSON", "ORGANIZATION", ..., "__NOVEL__"]},
          "subtype":     {"oneOf": [{"enum": ["POLITICIAN", ...]}, {"type": "null"}]},
          "description": {"type": "string"},
          "span":        {
            "type": "object",
            "required": ["start", "end"],
            "properties": {
              "start": {"type": "integer"},
              "end":   {"type": "integer"}
            }
          },
          "attributes": {
            "type": "object",
            // Union of all entity attribute keys across all types; all nullable.
            // System prompt carries per-type requirements (which are required/optional).
            "properties": {
              "nationality":         {"oneOf": [{"type": "string"}, {"type": "null"}]},
              "role_title":          {"oneOf": [{"type": "string"}, {"type": "null"}]},
              "date_of_birth":       {"oneOf": [{"type": "string"}, {"type": "null"}]},
              "jurisdiction":        {"oneOf": [{"type": "string"}, {"type": "null"}]},
              "registration_number": {"oneOf": [{"type": "string"}, {"type": "null"}]},
              "founding_date":       {"oneOf": [{"type": "string"}, {"type": "null"}]},
              "account_type":        {"oneOf": [{"type": "string"}, {"type": "null"}]},
              "institution":         {"oneOf": [{"type": "string"}, {"type": "null"}]},
              "currency":            {"oneOf": [{"type": "string"}, {"type": "null"}]},
              "country_code":        {"oneOf": [{"type": "string"}, {"type": "null"}]},
              "jurisdiction_type":   {"oneOf": [{"type": "string"}, {"type": "null"}]},
              "date":                {"oneOf": [{"type": "string"}, {"type": "null"}]},
              "event_type":          {"oneOf": [{"type": "string"}, {"type": "null"}]}
            }
          },
          "confidence":  {"type": "number", "minimum": 0, "maximum": 1}
        }
      }
    },
    "relations": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["head", "relation", "tail", "description", "evidence_span", "confidence"],
        "additionalProperties": false,
        "properties": {
          "head":          {"type": "string"},
          "relation":      {"enum": ["PAYMENT_TO", "OWNS", ..., "__UNCLASSIFIED__"]},
          "tail":          {"type": "string"},
          "description":   {"type": "string"},
          "evidence_span": {
            "type": "object",
            "required": ["start", "end"],
            "properties": {
              "start": {"type": "integer"},
              "end":   {"type": "integer"}
            }
          },
          "attributes": {
            "type": "object",
            "properties": {
              "amount":    {"oneOf": [{"type": "number"}, {"type": "null"}]},
              "currency":  {"oneOf": [{"type": "string"}, {"type": "null"}]},
              "date":      {"oneOf": [{"type": "string"}, {"type": "null"}]},
              "direction": {"oneOf": [{"type": "string"}, {"type": "null"}]}
            }
          },
          "confidence": {"type": "number", "minimum": 0, "maximum": 1}
        }
      }
    }
  }
}
```

Notes:
- Entity `attributes` is the union of all entity types' attribute keys; all nullable.
- Relation `attributes` is the union of all relation types' attribute keys; all nullable.
- Ollama constrained decoding guarantees structural validity (correct types, no unknown keys).
- Semantic validity (e.g. `amount` filled for PAYMENT_TO, `jurisdiction` for ORGANIZATION)
  is enforced via the system prompt and checked post-hoc: missing `required: true` attributes
  reduce confidence and raise a review flag. The LLM is never forced to hallucinate a value.
- `system_prompt()` generates a human-readable attribute requirements table injected into
  the extraction system prompt:

  ```
  ENTITY ATTRIBUTE REQUIREMENTS
  ==============================
  PERSON         → role_title (optional), nationality (optional), date_of_birth (optional)
  ORGANIZATION   → jurisdiction (optional), registration_number (optional), founding_date (optional)
  FINANCIAL_ENTITY → account_type (optional), institution (optional), currency (optional)
  ...

  RELATION ATTRIBUTE REQUIREMENTS
  ================================
  PAYMENT_TO     → amount (REQUIRED), currency (REQUIRED), date (optional), direction (optional)
  OWNS           → stake_pct (optional), date (optional)
  ...
  ```

### `weaviate_collection_spec()` output structure

```jsonc
{
  "name": "Entity_financial_flows_v1",
  "description": "Canonical entities for use case financial_flows, schema v1",
  "vectorizer_config": [{"vectorizer": {"none": {}}}],
  "properties": [
    {"name": "canonical_name", "dataType": ["text"]},
    {"name": "type",           "dataType": ["text"]},
    {"name": "subtype",        "dataType": ["text"]},
    {"name": "description",    "dataType": ["text"]},
    {"name": "aliases",        "dataType": ["text[]"]},
    {"name": "doc_ids",        "dataType": ["text[]"]},
    {"name": "schema_version", "dataType": ["text"]}
  ],
  "vectorIndexConfig": {"distance": "cosine"}
}
```

The collection name embeds the use case and schema version so incompatible schemas never
share a collection. Migrations create a new collection, backfill, then reroute writes.

---

## Implementation notes

### Caching

```python
_cache: dict[str, Ontology] = {}

def load(use_case: str) -> Ontology:
    if use_case not in _cache:
        path = _ONTOLOGY_DIR / f"{use_case}.yaml"
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

## Dependencies

- `pyyaml` (already in requirements.txt)
- No new model or network call
