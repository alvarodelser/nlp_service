# Ontology Loader — Design & Implementation

## Purpose

Single source of truth for a use-case's entity types, relation types, and their constraints.
Every module that touches the extraction pipeline imports the ontology rather than hardcoding
types. The loader validates the YAML at startup, then emits two derived artefacts on demand:

- A **JSON Schema** consumed by Ollama's constrained decoding (`format=`) in Pass 1
- A **Weaviate collection spec** consumed by persistence when creating or migrating a collection

One process may load multiple ontologies (one per active use case).

---

## File layout

```
config/
  ontologies/
    financial_flows.yaml      ← first use case (journalism / payments)
    <next_use_case>.yaml

nlp/
  ontology.py                 ← loader, validator, schema emitters
```

No router or endpoint. Other modules call `nlp.ontology.load(use_case)`.

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
    subtypes:
      - name: POLITICIAN
        description: Elected or appointed public official.
      - name: EXECUTIVE
        description: Senior corporate officer (CEO, CFO, board member).
      - name: INTERMEDIARY
        description: Lawyer, accountant, nominee director, or agent acting
                     on behalf of another party.

  - name: ORGANIZATION
    description: >
      Any formal or informal group acting as a unit: company, fund,
      foundation, government body, or criminal network.
    subtypes:
      - name: SHELL_COMPANY
        description: Legal entity with no active business operations,
                     typically used to hold assets or obscure ownership.
      - name: BANK
        description: Licensed deposit-taking institution.
      - name: FUND
        description: Investment vehicle (hedge fund, PE fund, trust).
      - name: GOVERNMENT_BODY
        description: Regulatory authority, ministry, or state-owned enterprise.

  - name: FINANCIAL_ENTITY
    description: >
      An account, instrument, or financial product: bank account, wire
      transfer, bond, loan, equity stake.
    subtypes: []

  - name: LOCATION
    description: >
      A jurisdiction, country, city, address, or offshore zone relevant
      to where entities are registered or transactions occur.
    subtypes: []

  - name: EVENT
    description: >
      A dateable occurrence: a meeting, a transaction, a filing, a
      court ruling.
    subtypes: []

relation_types:
  - name: PAYMENT_TO
    description: >
      One party transfers money or assets to another. Capture
      amount, currency, date, and mechanism (wire, cash, crypto) in
      attributes.
    head_types: [PERSON, ORGANIZATION, FINANCIAL_ENTITY]
    tail_types: [PERSON, ORGANIZATION, FINANCIAL_ENTITY]
    attributes:
      amount:    {type: number, description: Numeric amount, no currency symbol}
      currency:  {type: string, description: ISO 4217 code (USD, EUR…)}
      date:      {type: string, description: ISO 8601 date or partial (2024-Q1)}
      direction: {type: string, description: "incoming | outgoing | unknown"}

  - name: OWNS
    description: One party holds an ownership stake in another.
    head_types: [PERSON, ORGANIZATION]
    tail_types: [ORGANIZATION, FINANCIAL_ENTITY]
    attributes:
      stake_pct: {type: number, description: Ownership percentage if stated}
      date:      {type: string}

  - name: CONTROLS
    description: >
      De-facto control without necessarily owning a majority stake.
    head_types: [PERSON, ORGANIZATION]
    tail_types: [ORGANIZATION]
    attributes: {}

  - name: MEMBER_OF
    description: A person holds a formal role inside an organisation.
    head_types: [PERSON]
    tail_types: [ORGANIZATION]
    attributes:
      role:      {type: string, description: Job title or board position}
      date_from: {type: string}
      date_to:   {type: string}

  - name: REGISTERED_IN
    description: An entity is legally incorporated or registered in a jurisdiction.
    head_types: [ORGANIZATION, FINANCIAL_ENTITY]
    tail_types: [LOCATION]
    attributes:
      date: {type: string}

  - name: PARTY_TO
    description: A person or organisation is a named party to an event (deal, lawsuit, filing).
    head_types: [PERSON, ORGANIZATION]
    tail_types: [EVENT]
    attributes:
      role: {type: string, description: "plaintiff | defendant | signatory | witness | other"}
```

### Validation rules (enforced at load time)

| Rule | Check |
|---|---|
| `name` is SCREAMING_SNAKE_CASE | regex |
| All `head_types` / `tail_types` values are declared entity type names | set membership |
| Each `attributes` entry has `type` in `{string, number, boolean}` | enum |
| `version` is a string that can be compared with `packaging.version` | cast |
| No duplicate names within `entity_types` or `relation_types` | set |

---

## Python API

```python
from nlp.ontology import load, Ontology

onto: Ontology = load("financial_flows")  # cached after first call

onto.entity_type_names()          # -> ["PERSON", "ORGANIZATION", ...]
onto.subtype_names()              # -> ["POLITICIAN", "EXECUTIVE", ...] (all, pooled)
onto.relation_type_names()        # -> ["PAYMENT_TO", "OWNS", ...]
onto.relation_type("PAYMENT_TO")  # -> RelationType(...)
onto.validate_subtype("POLITICIAN", parent="PERSON")  # -> True

onto.extraction_schema()          # -> dict  (JSON Schema for Ollama format=)
onto.system_prompt()              # -> str   (ontology descriptions for LLM system message)
onto.weaviate_collection_spec()   # -> dict  (Weaviate collection definition)
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
          "type":        {"enum": ["PERSON", "ORGANIZATION", ...]},
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
          "relation":      {"enum": ["PAYMENT_TO", "OWNS", ...]},
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

Note: `attributes` properties are derived from the union of all relation types' attribute
definitions. Ollama constrained decoding guarantees structural validity; semantic validity
(e.g. amount present only on PAYMENT_TO) is checked post-hoc in the extractor.

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
