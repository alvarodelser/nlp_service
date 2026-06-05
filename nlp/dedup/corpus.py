# nlp_service/nlp/dedup/corpus.py
"""Whole-collection clustering: union-find over near-neighbour edges (auto-merge above the tight
threshold, LLM double-check in the mid band)."""
from .llm import adjudicate


class _UnionFind:
    def __init__(self, ids) -> None:
        self.parent = {i: i for i in ids}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def groups(self) -> list[list[str]]:
        out: dict[str, list[str]] = {}
        for i in self.parent:
            out.setdefault(self.find(i), []).append(i)
        return list(out.values())


def cluster(index, *, kind, compare_property, merge_threshold, llm_low, top_k,
            type_filter=None, llm=True) -> list[list[str]]:
    rp = (compare_property,) + (("type",) if type_filter else ())
    objs = [o for o in index.fetch_all(return_props=rp, with_vector=True)
            if not type_filter or o.props.get("type") == type_filter]
    uf = _UnionFind([o.id for o in objs])

    for o in objs:
        for cand in index.near(o.vector, top_k, type_filter=type_filter,
                               return_props=(compare_property,)):
            if cand.id == o.id:
                continue
            if cand.score >= merge_threshold:
                uf.union(o.id, cand.id)
            elif cand.score >= llm_low and llm:
                if adjudicate(kind, o.props.get(compare_property, ""),
                              cand.props.get(compare_property, "")) == "yes":
                    uf.union(o.id, cand.id)
    return uf.groups()
