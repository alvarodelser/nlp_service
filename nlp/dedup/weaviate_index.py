# nlp_service/nlp/dedup/weaviate_index.py
"""Read-only access to one Weaviate collection over HTTP (REST + GraphQL) via httpx — the same
'httpx -> sidecar on b4c-net' pattern used for Ollama/the vectorizer. No weaviate-client, no gRPC.
The orchestrators own all writes."""
import os
from dataclasses import dataclass, field

import httpx

WEAVIATE_URL     = os.environ.get("WEAVIATE_URL", "http://weaviate:8080").rstrip("/")
WEAVIATE_TIMEOUT = float(os.environ.get("WEAVIATE_TIMEOUT", "30"))


@dataclass
class Candidate:
    id:    str
    score: float                 # cosine similarity (1 - distance)
    props: dict


@dataclass
class FetchedObject:
    id:     str
    vector: list[float]
    props:  dict = field(default_factory=dict)


class WeaviateIndex:
    def __init__(self, collection: str, base_url: str | None = None, client=None) -> None:
        self.collection = collection
        self._base = (base_url or WEAVIATE_URL).rstrip("/")
        self._client = client or httpx.Client(timeout=WEAVIATE_TIMEOUT)

    def _graphql(self, query: str) -> list[dict]:
        r = self._client.post(f"{self._base}/v1/graphql", json={"query": query})
        r.raise_for_status()
        body = r.json()
        if body.get("errors"):
            raise RuntimeError(f"weaviate graphql error: {body['errors']}")
        return body["data"]["Get"][self.collection] or []

    @staticmethod
    def _where(type_filter) -> str:
        return (f', where: {{path:["type"], operator:Equal, valueText:"{type_filter}"}}'
                if type_filter else "")

    def near(self, embedding, top_k, type_filter=None, return_props=()) -> list[Candidate]:
        props = " ".join(return_props)
        vec = ",".join(repr(float(x)) for x in embedding)
        q = (f'{{ Get {{ {self.collection}('
             f'nearVector: {{vector: [{vec}]}}, limit: {top_k}{self._where(type_filter)}) '
             f'{{ {props} _additional {{ id distance }} }} }} }}')
        return [Candidate(id=o["_additional"]["id"],
                          score=1.0 - o["_additional"]["distance"],
                          props={k: o.get(k) for k in return_props})
                for o in self._graphql(q)]

    def fetch_all(self, return_props=(), with_vector=True, page=200):
        props = " ".join(return_props)
        add = "id vector" if with_vector else "id"
        after = None
        while True:
            cursor = f', after: "{after}"' if after else ""
            q = (f'{{ Get {{ {self.collection}(limit: {page}{cursor}) '
                 f'{{ {props} _additional {{ {add} }} }} }} }}')
            objs = self._graphql(q)
            if not objs:
                break
            for o in objs:
                yield FetchedObject(id=o["_additional"]["id"],
                                    vector=o["_additional"].get("vector") or [],
                                    props={k: o.get(k) for k in return_props})
            after = objs[-1]["_additional"]["id"]
            if len(objs) < page:
                break
