# nlp_service/nlp/ner/service.py
import logging

from . import ollama_client
from .schema_compiler import compile_schema
from .schema_types import (ExtractedEntity, ExtractedRelation, ExtractionSchema, NerResponse)

log = logging.getLogger(__name__)


def _check_rules(raw: dict, schema: ExtractionSchema) -> NerResponse:
    """Grammar guarantees valid types/subtypes/attributes/datatypes. The only runtime check is
    the one thing the grammar cannot express: relation endpoint type compatibility."""
    entities = [ExtractedEntity(**e) for e in raw.get("entities", [])]
    rel_types = {rt.name: (set(rt.head_types), set(rt.tail_types))
                 for rt in schema.relation_types}

    relations: list[ExtractedRelation] = []
    for r in raw.get("relations", []):
        head, tail = r["head"], r["tail"]
        if not (0 <= head < len(entities) and 0 <= tail < len(entities)):
            log.warning("dropping relation %r: endpoint index out of range", r.get("type"))
            continue
        heads, tails = rel_types.get(r["type"], (set(), set()))
        if entities[head].type not in heads or entities[tail].type not in tails:
            log.warning("dropping relation %r: endpoint types incompatible", r.get("type"))
            continue
        relations.append(ExtractedRelation(**r))

    return NerResponse(entities=entities, relations=relations)


def run(text: str, schema: ExtractionSchema) -> NerResponse:
    """Stateless extraction: text + schema -> entities + relations.

    Raises httpx.HTTPError if Ollama is unavailable (after retries).
    """
    grammar, prompt = compile_schema(schema)
    raw = ollama_client.extract(system=prompt, user=text, grammar=grammar)
    return _check_rules(raw, schema)
