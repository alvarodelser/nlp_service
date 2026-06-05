# nlp_service/nlp/resolver/premerge.py
"""Pure-Python exact-name blocking: collapse entities sharing a normalized name + type into one
candidate. No LLM."""
import re
import unicodedata
from dataclasses import dataclass, field

from .types import EntityIn

_LEGAL = {"sa", "sl", "sau", "slu", "ltd", "inc", "gmbh", "plc", "llc", "sas", "bv", "srl", "co"}
_HONORIFIC = {"sr", "sra", "srta", "dr", "dra", "don", "dona", "d", "da", "mr", "mrs", "ms"}


@dataclass
class Candidate:
    canonical_name: str            # most complete (longest) member name
    names:          list[str]      # member surface names (deduped, order-preserved)
    type:           str
    subtype:        str | None
    evidence:       list[str] = field(default_factory=list)


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def normalize_name(name: str) -> str:
    s = _strip_accents(name).lower()
    s = re.sub(r"[.,;:\"'()\[\]]", " ", s)
    toks = [t for t in s.split() if t and t not in _HONORIFIC and t not in _LEGAL]
    return " ".join(toks)


def premerge(entities: list[EntityIn]) -> list[Candidate]:
    groups: dict[tuple[str, str], Candidate] = {}
    order: list[tuple[str, str]] = []
    for e in entities:
        key = (normalize_name(e.name), e.type)
        c = groups.get(key)
        if c is None:
            groups[key] = Candidate(canonical_name=e.name, names=[e.name], type=e.type,
                                    subtype=e.subtype, evidence=[e.evidence])
            order.append(key)
        else:
            if e.name not in c.names:
                c.names.append(e.name)
            c.evidence.append(e.evidence)
            if len(e.name) > len(c.canonical_name):
                c.canonical_name = e.name
            if c.subtype is None and e.subtype:
                c.subtype = e.subtype
    return [groups[k] for k in order]
