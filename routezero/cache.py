from dataclasses import dataclass, field

import chromadb
import numpy as np
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
        self._client = None
        self._chroma_available = True
        self._memory_entries: list[dict] = []
        try:
            self._client = chromadb.PersistentClient(path=settings.chromadb_path)
        except Exception:
            self._chroma_available = False
        self.embedder = SentenceTransformer(settings.embedding_model)
        self._hit_count: int = 0
        self._query_count: int = 0

    def _get_collection(self) -> chromadb.Collection:
        if self._client is None or not self._chroma_available:
            raise RuntimeError("ChromaDB cache is unavailable")
        return self._client.get_or_create_collection(
            name="routezero_cache",
            metadata={"hnsw:space": "cosine"},
        )

    def embed(self, text: str) -> list[float]:
        embedding = self.embedder.encode(text)
        return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

    def _query_memory(self, embedding: list[float]) -> CacheHitResult | None:
        if not self._memory_entries:
            return None

        query_vec = np.array(embedding, dtype=float)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return None

        best_entry: dict | None = None
        best_similarity = -1.0
        for entry in self._memory_entries:
            stored_vec = np.array(entry["embedding"], dtype=float)
            stored_norm = np.linalg.norm(stored_vec)
            if stored_norm == 0:
                continue
            similarity = float(np.dot(query_vec, stored_vec) / (query_norm * stored_norm))
            if similarity > best_similarity:
                best_similarity = similarity
                best_entry = entry

        if best_entry is None or best_similarity < self.settings.cache_similarity_threshold:
            return None

        self._hit_count += 1
        return CacheHitResult(
            response=best_entry["response"],
            similarity=best_similarity,
            task_type=best_entry["metadata"].get("task_type", ""),
        )

    def query(self, prompt: str) -> CacheHitResult | None:
        self._query_count += 1
        embedding = self.embed(prompt)
        try:
            collection = self._get_collection()
            results = collection.query(
                query_embeddings=[embedding],
                n_results=1,
                include=["documents", "distances", "metadatas"],
            )
        except Exception:
            self._chroma_available = False
            return self._query_memory(embedding)

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
        doc_id = str(hash(prompt))
        if self._chroma_available:
            try:
                collection = self._get_collection()
                collection.add(
                    embeddings=[embedding],
                    documents=[response],
                    metadatas=[metadata],
                    ids=[doc_id],
                )
                return
            except Exception:
                self._chroma_available = False

        self._memory_entries = [
            entry for entry in self._memory_entries if entry["id"] != doc_id
        ]
        self._memory_entries.append(
            {
                "id": doc_id,
                "embedding": embedding,
                "response": response,
                "metadata": metadata,
            }
        )

    def clear(self) -> None:
        """Delete all entries from the cache collection."""
        self._memory_entries.clear()
        try:
            self._client.delete_collection("routezero_cache")
        except Exception:
            self._chroma_available = False
            pass  # collection may not exist yet

    def stats(self) -> CacheStats:
        try:
            collection = self._get_collection()
            total_entries = collection.count()
        except Exception:
            self._chroma_available = False
            total_entries = len(self._memory_entries)
        hit_ratio = (
            self._hit_count / self._query_count if self._query_count > 0 else 0.0
        )
        return CacheStats(total_entries=total_entries, hit_ratio=hit_ratio)
