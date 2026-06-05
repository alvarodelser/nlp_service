# Extraction Schema (Type Catalogue) — Design & Implementation

## Purpose

Single source of truth for a use-case's entity types, relation types, their extractable
attributes, and visual encoding hints. This is **orchestrator-side config**: the orchestrator
loads and validates it, then uses it to drive the pipeline. It is *not* on the extractor's
critical path — the extractor is stateless and receives a schema in its request (see 02).

The loader validates the YAML at startup, then emits two derived artefacts on demand:

- An **`ExtractionSchema`** — the slim, inline type description handed to the entity extractor
  (02). The extractor compiles it into the LLM grammar and prompt; this module builds neither.
- A **Weaviate collection spec** — consumed by persistence when creating or migrating the
  entity collection (one column per declared attribute).

One process may load multiple schemas (one per active use case).

The schema is **closed-world**: the LLM may only emit entity and relation types declared in
the YAML. Constrained decoding makes any other type literally unrepresentable. Text that
matches no declared type is simply not extracted — there is no `__NOVEL__`/`__UNCLASSIFIED__`
escape hatch. Growing the catalogue is a deliberate, versioned edit to the YAML.

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

No router, no endpoint, no LLM, no network. The orchestrator calls `nlp.schema.load(use_case)`
and passes the projected `ExtractionSchema` to the extractor with each request.

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

schema.entity_type_names()              # -> ["PERSON", "ORGANIZATION", "FINANCIAL_ENTITY", "LOCATION", "EVENT"]
schema.subtype_names()                  # -> ["POLITICIAN", "EXECUTIVE", ...] (all, pooled)
schema.relation_type_names()            # -> ["PAYMENT_TO", "OWNS", "CONTROLS", "MEMBER_OF", "REGISTERED_IN", "PARTY_TO"]
schema.subtypes_for("PERSON")           # -> ["POLITICIAN", "EXECUTIVE", "INTERMEDIARY"] (per type, for discriminated branches)
schema.entity_attrs("PERSON")           # -> ["nationality", "role_title", "date_of_birth"] (per type, for discriminated branches)
schema.relation_attrs("PAYMENT_TO")     # -> ["amount", "currency", "date", "direction"]
schema.entity_type("ORGANIZATION")      # -> EntityType(...)
schema.relation_type("PAYMENT_TO")      # -> RelationType(...)
schema.validate_subtype("POLITICIAN", parent="PERSON")  # -> True

# All entity attribute keys across all types (union — for the permissive response model only;
# the projected ExtractionSchema scopes attributes per type)
schema.all_entity_attribute_keys()      # -> ["nationality", "role_title", "jurisdiction", ...]
schema.all_relation_attribute_keys()    # -> ["amount", "currency", "date", "stake_pct", ...]

# The two derived artefacts:
schema.to_extraction_schema()      # -> ExtractionSchema  (slim inline contract for the extractor, see 02)
schema.weaviate_collection_spec()  # -> dict              (Weaviate collection definition)
```

### `to_extraction_schema()` output

Returns an `ExtractionSchema` — the Pydantic contract defined in 02 — a faithful projection of
the YAML into exactly what the extractor needs, and nothing more:

- **entity types**: `name`, `description`, `subtypes` (name + description), `attributes`
  (name + `datatype`)
- **relation types**: the same, plus `head_types` / `tail_types`

```python
ExtractionSchema(
    entity_types=[
        EntityTypeDef(name="PERSON", description="...",
                      subtypes=[SubtypeDef("POLITICIAN", "..."), SubtypeDef("EXECUTIVE", "..."), ...],
                      attributes=[AttributeDef("nationality", "string", "..."),
                                  AttributeDef("role_title", "string", "..."), ...]),
        EntityTypeDef(name="ORGANIZATION", ...),
        # ... one per declared entity type
    ],
    relation_types=[
        RelationTypeDef(name="PAYMENT_TO", description="...",
                        head_types=["PERSON", "ORGANIZATION", "FINANCIAL_ENTITY"],
                        tail_types=["PERSON", "ORGANIZATION", "FINANCIAL_ENTITY"],
                        attributes=[AttributeDef("amount", "number", "..."),
                                    AttributeDef("currency", "string", "..."), ...]),
        # ... one per declared relation type
    ],
)
```

Dropped in the projection: `visual` hints, `required` flags, and `version` — none of which the
extractor needs. The extractor (02 §1) compiles this into the discriminated-union grammar and
the prompt; **this module builds neither grammar nor prompt.** Because the catalogue is
closed-world, the grammar the extractor derives can only ever emit the declared types.

### `weaviate_collection_spec()` output structure

```jsonc
{
  "name": "Entity_financial_flows_v1",
  "description": "Canonical entities for use case financial_flows, schema v1",
  "vectorizer_config": [{"vectorizer": {"none": {}}}],
  "properties": [
    // --- fixed core fields (always present) ---
    {"name": "canonical_name", "dataType": ["text"]},
    {"name": "type",           "dataType": ["text"]},
    {"name": "subtype",        "dataType": ["text"]},
    {"name": "description",    "dataType": ["text"]},   // Spanish
    {"name": "aliases",        "dataType": ["text[]"]}, // surface forms from mentions
    {"name": "doc_ids",        "dataType": ["text[]"]},
    {"name": "schema_version", "dataType": ["text"]},

    // --- one property per declared entity attribute (union across all entity types) ---
    // dataType is mapped from the YAML attribute type: string→text, number→number, boolean→boolean.
    // This is what gives every extracted attribute a place to land at upsert.
    {"name": "nationality",         "dataType": ["text"]},
    {"name": "role_title",          "dataType": ["text"]},
    {"name": "date_of_birth",       "dataType": ["text"]},
    {"name": "jurisdiction",        "dataType": ["text"]},
    {"name": "registration_number", "dataType": ["text"]},
    {"name": "founding_date",       "dataType": ["text"]},
    {"name": "account_type",        "dataType": ["text"]},
    {"name": "institution",         "dataType": ["text"]},
    {"name": "currency",            "dataType": ["text"]},
    {"name": "country_code",        "dataType": ["text"]},
    {"name": "jurisdiction_type",   "dataType": ["text"]},
    {"name": "date",                "dataType": ["text"]},
    {"name": "event_type",          "dataType": ["text"]}
  ],
  "vectorIndexConfig": {"distance": "cosine"}
}
```

The attribute properties are generated from the schema (`all_entity_attribute_keys()` for the
names, each attribute's `type` for the `dataType`) — so adding an attribute to the YAML adds a
column here automatically. The orchestrator populates them at upsert from each cluster's merged
`ExtractedEntity.attributes`. The collection name embeds the use case and schema version so
incompatible schemas never share a collection; migrations create a new collection, backfill,
then reroute writes.

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

(No post-extraction subtype validation is needed: the extractor's discriminated-union grammar
only allows a subtype that belongs to its parent type, so a mismatch can't be produced.)

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
