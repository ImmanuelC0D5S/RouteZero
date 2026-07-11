from dataclasses import dataclass, field
import logging
import os
import chromadb
from sentence_transformers import SentenceTransformer
from routezero.config import Settings

logger = logging.getLogger(__name__)

@dataclass
class CacheHitResult:
    response: str
    similarity: float
    task_type: str

class SemanticCache:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        
        # FIX 1: Use the local path if it exists to avoid downloading from HF
        # Your Dockerfile copies model_cache/ into the root
        local_model_path = "./model_cache/all-MiniLM-L6-v2"
        model_to_load = local_model_path if os.path.exists(local_model_path) else settings.embedding_model
        
        logger.info(f"Loading embedding model from: {model_to_load}")
        self.embedder = SentenceTransformer(model_to_load, device="cpu")
        
        # FIX 2: Make ChromaDB initialization robust
        # If we can't write to the path, we use EphemeralClient (RAM only) 
        # so the app doesn't crash.
        try:
            self._client = chromadb.PersistentClient(path=settings.chromadb_path)
        except Exception as e:
            logger.warning(f"Could not init PersistentClient: {e}. Falling back to Ephemeral.")
            self._client = chromadb.EphemeralClient()

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
        """
        Note: This is bypassed in pipeline.py for compliance, 
        but logic is kept here for structural integrity.
        """
        try:
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
        except Exception as e:
            logger.error(f"Cache query failed: {e}")
        return None

    def insert(self, prompt: str, response: str, metadata: dict) -> None:
        """
        Note: This is commented out in pipeline.py for compliance.
        """
        try:
            embedding = self.embed(prompt)
            collection = self._get_collection()
            doc_id = str(hash(prompt))
            collection.add(
                embeddings=[embedding],
                documents=[response],
                metadatas=[metadata],
                ids=[doc_id],
            )
        except Exception as e:
            logger.error(f"Cache insert failed: {e}")

    # ... keep clear() and stats() as they are