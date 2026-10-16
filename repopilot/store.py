"""Qdrant stores vectors plus source payloads; Python always scopes retrieval."""

import hashlib
import os
from uuid import uuid4

from qdrant_client import QdrantClient, models

from repopilot.rag_models import Chunk, Evidence


class StoreUnavailable(RuntimeError):
    pass


def repository_key(url: str) -> str:
    from repopilot.github import parse_repository_url
    return "/".join(parse_repository_url(url)).lower()


def alias_for(repository: str) -> str:
    return "repo_" + hashlib.sha256(repository.encode()).hexdigest()[:24]


class VectorStore:
    def __init__(self, client=None):
        self.client = client or QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"), timeout=30)

    def replace(self, repository: str, chunks: list[Chunk], vectors: list[list[float]]):
        if not chunks or len(chunks) != len(vectors) or any(chunk.repository != repository for chunk in chunks):
            raise ValueError("Index must contain matching vectors and chunks from one repository.")
        alias = alias_for(repository)
        collection = alias + "_" + uuid4().hex[:12]
        try:
            previous = next((a.collection_name for a in self.client.get_aliases().aliases if a.alias_name == alias), None)
            self.client.create_collection(collection, vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE))
            # Upload to a new collection first. A failed upload never becomes visible.
            for start in range(0, len(chunks), 64):
                self.client.upsert(collection, points=[models.PointStruct(
                    id=chunk.chunk_id, vector=vector, payload=chunk.model_dump(),
                ) for chunk, vector in zip(chunks[start:start+64], vectors[start:start+64])], wait=True)
            operations = []
            if previous:
                operations.append(models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias)))
            operations.append(models.CreateAliasOperation(create_alias=models.CreateAlias(collection_name=collection, alias_name=alias)))
            # Qdrant applies this alias switch atomically; queries see old or new data.
            self.client.update_collection_aliases(change_aliases_operations=operations)
        except Exception as error:
            # Preserve the old published alias. Orphan staging data can be removed later.
            raise StoreUnavailable("Qdrant indexing failed; completion could not be confirmed. Retry indexing.") from error
        # Cleanup is best effort AFTER publication, never part of its success condition.
        if previous:
            try:
                self.client.delete_collection(previous)
            except Exception:
                pass

    def search(self, repository: str, vector: list[float], top_k=5, document_type=None) -> list[Evidence]:
        alias = alias_for(repository)
        try:
            if not self.client.collection_exists(alias):
                raise ValueError("Repository is not indexed. Index it before asking a question.")
            conditions = [models.FieldCondition(key="repository", match=models.MatchValue(value=repository))]
            if document_type:
                conditions.append(models.FieldCondition(key="document_type", match=models.MatchValue(value=document_type)))
            points = self.client.query_points(
                collection_name=alias, query=vector, query_filter=models.Filter(must=conditions),
                limit=top_k, with_payload=True,
            ).points
            results = [Evidence(chunk=Chunk.model_validate(point.payload), score=point.score) for point in points]
            if any(item.chunk.repository != repository for item in results):
                raise RuntimeError("Repository isolation check failed.")
            return results
        except ValueError:
            raise
        except Exception as error:
            raise StoreUnavailable("Qdrant retrieval failed. Check that Qdrant is running.") from error
