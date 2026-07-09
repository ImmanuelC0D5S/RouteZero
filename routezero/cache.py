from dataclasses import dataclass, field

import chromadb
from sentence_transformers import SentenceTransformer

from routezero.config import Settings


@dataclass
class CacheHitResult:
    response: str
    similarity: float
    task_type: str


@dataclass
class CacheStats:
    total_entries: int
    hit_ratio: float


class SemanticCache:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = chromadb.PersistentClient(path=settings.chromadb_path)
        self.embedder = SentenceTransformer(settings.embedding_model)
        self._hit_count: int = 0
        self._query_count: int = 0

    def _get_collection(self) -> chromadb.Collection:
        return self._client.get_or_create_collection(
            name="routezero_cache",
            metadata={"hnsw:space": "cosine"},
        )

    def embed(self, text: str) -> list[float]:
        return self.embedder.encode(text).tolist()

    def query(self, prompt: str) -> CacheHitResult | None:
        self._query_count += 1
        embedding = self.embed(prompt)
        collection = self._get_collection()
        results = collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["documents", "distances", "metadatas"],
        )

        if not results["distances"] or not results["distances"][0]:
            return None

        distance = results["distances"][0][0]
        similarity = 1.0 - distance / 2.0

        if similarity >= self.settings.cache_similarity_threshold:
            self._hit_count += 1
            document = results["documents"][0][0]
            metadata = results["metadatas"][0][0]
            return CacheHitResult(
                response=document,
                similarity=similarity,
                task_type=metadata.get("task_type", ""),
            )

        return None

    def insert(self, prompt: str, response: str, metadata: dict) -> None:
        embedding = self.embed(prompt)
        collection = self._get_collection()
        doc_id = str(hash(prompt))
        collection.add(
            embeddings=[embedding],
            documents=[response],
            metadatas=[metadata],
            ids=[doc_id],
        )

    def clear(self) -> None:
        """Delete all entries from the cache collection."""
        try:
            self._client.delete_collection("routezero_cache")
        except Exception:
            pass  # collection may not exist yet

    def stats(self) -> CacheStats:
        collection = self._get_collection()
        total_entries = collection.count()
        hit_ratio = (
            self._hit_count / self._query_count if self._query_count > 0 else 0.0
        )
        return CacheStats(total_entries=total_entries, hit_ratio=hit_ratio)
