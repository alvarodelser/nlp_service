"""Write-capable access to the relation pipeline's Weaviate collections over HTTP (REST) via httpx
— the same pattern as the news pipeline. Client owns the working (per-document) entity + relation
collections; FinalClient owns the consolidated collections. Reads for Phase B clustering go
through the module's own dedup.WeaviateIndex; this class owns the writes and point gets."""
import os

import httpx

from . import ontology

WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://weaviate:8080").rstrip("/")
WEAVIATE_TIMEOUT = float(os.environ.get("WEAVIATE_TIMEOUT", "30"))


class _PairClient:
    """Entity + relation collection pair for one use case (working or final)."""

    def __init__(self, use_case: str, final: bool, base_url: str | None = None, client=None) -> None:
        self.use_case = use_case
        self.entity_collection = ontology.entity_collection_name(use_case, final)
        self.relation_collection = ontology.relation_collection_name(use_case, final)
        self._entity_spec = ontology.weaviate_entity_spec(use_case, final)
        self._relation_spec = ontology.weaviate_relation_spec(use_case, final)
        base = (base_url or WEAVIATE_URL).rstrip("/")
        self._client = client or httpx.Client(base_url=base, timeout=WEAVIATE_TIMEOUT)

    def _ensure_one(self, spec: dict) -> None:
        if self._client.get(f"/v1/schema/{spec['class']}").status_code == 200:
            return
        self._client.post("/v1/schema", json=spec).raise_for_status()

    def ensure_collections(self) -> None:
        self._ensure_one(self._entity_spec)
        self._ensure_one(self._relation_spec)

    def _upsert(self, collection: str, properties: dict, vector) -> str:
        r = self._client.post("/v1/objects", json={
            "class": collection, "properties": properties, "vector": vector})
        r.raise_for_status()
        return r.json()["id"]

    def _get(self, collection: str, obj_id: str) -> dict:
        r = self._client.get(f"/v1/objects/{collection}/{obj_id}")
        r.raise_for_status()
        return r.json().get("properties", {})

    def upsert_entity(self, properties: dict, vector) -> str:
        return self._upsert(self.entity_collection, properties, vector)

    def upsert_relation(self, properties: dict, vector) -> str:
        return self._upsert(self.relation_collection, properties, vector)

    def get_entity(self, obj_id: str) -> dict:
        return self._get(self.entity_collection, obj_id)

    def get_relation(self, obj_id: str) -> dict:
        return self._get(self.relation_collection, obj_id)


class Client(_PairClient):
    def __init__(self, use_case: str, base_url: str | None = None, client=None) -> None:
        super().__init__(use_case, final=False, base_url=base_url, client=client)


class FinalClient(_PairClient):
    def __init__(self, use_case: str, base_url: str | None = None, client=None) -> None:
        super().__init__(use_case, final=True, base_url=base_url, client=client)
