"""Write-capable access to the news article collection over Weaviate's HTTP API (REST + GraphQL)
via httpx — the same 'httpx -> sidecar' pattern used elsewhere, no weaviate-client, no gRPC. The
read-only nlp.dedup.WeaviateIndex is left untouched; the orchestrator owns all writes here."""
import os
from dataclasses import dataclass, field

import httpx

WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://weaviate:8080").rstrip("/")
WEAVIATE_TIMEOUT = float(os.environ.get("WEAVIATE_TIMEOUT", "30"))


@dataclass
class RecentObject:
    uuid: str
    properties: dict = field(default_factory=dict)


def _class_definition(collection: str) -> dict:
    return {
        "class": collection,
        "vectorizer": "none",
        "properties": [
            {"name": "headline", "dataType": ["text"]},
            {"name": "summary", "dataType": ["text"]},
            {"name": "sources", "dataType": ["text[]"]},
            {"name": "urls", "dataType": ["text[]"]},
            {"name": "date", "dataType": ["date"]},
            {"name": "minhash", "dataType": ["int[]"]},
            {"name": "topics", "dataType": ["text[]"]},
            {"name": "scope", "dataType": ["text"]},
            {"name": "region", "dataType": ["text"]},
            {"name": "city_ids", "dataType": ["int[]"]},
            {"name": "places", "dataType": ["text"]},
        ],
        "vectorIndexConfig": {"distance": "cosine"},
    }


class Client:
    def __init__(self, collection: str, base_url: str | None = None, client=None) -> None:
        self.collection = collection
        base = (base_url or WEAVIATE_URL).rstrip("/")
        self._client = client or httpx.Client(base_url=base, timeout=WEAVIATE_TIMEOUT)

    # --- schema ---

    def ensure_collection(self) -> None:
        r = self._client.get(f"/v1/schema/{self.collection}")
        if r.status_code == 200:
            return
        create = self._client.post("/v1/schema", json=_class_definition(self.collection))
        create.raise_for_status()

    # --- object I/O ---

    def get(self, obj_id: str, return_props=()) -> dict:
        r = self._client.get(f"/v1/objects/{self.collection}/{obj_id}")
        r.raise_for_status()
        props = r.json().get("properties", {})
        return {k: props.get(k) for k in return_props}

    def upsert(self, properties: dict, vector: list[float]) -> str:
        r = self._client.post("/v1/objects", json={
            "class": self.collection, "properties": properties, "vector": vector,
        })
        r.raise_for_status()
        return r.json()["id"]

    def update(self, obj_id: str, properties: dict) -> None:
        r = self._client.patch(f"/v1/objects/{self.collection}/{obj_id}",
                               json={"class": self.collection, "properties": properties})
        r.raise_for_status()

    # --- candidate window (minhash dedup) ---

    def iter_recent(self, return_props=(), limit=5000):
        props = " ".join(return_props)
        q = (f'{{ Get {{ {self.collection}('
             f'limit: {int(limit)}, sort: [{{path: ["date"], order: desc}}]) '
             f'{{ {props} _additional {{ id }} }} }} }}')
        r = self._client.post("/v1/graphql", json={"query": q})
        r.raise_for_status()
        body = r.json()
        if body.get("errors"):
            raise RuntimeError(f"weaviate graphql error: {body['errors']}")
        for o in body["data"]["Get"][self.collection] or []:
            yield RecentObject(uuid=o["_additional"]["id"],
                               properties={k: o.get(k) for k in return_props})
