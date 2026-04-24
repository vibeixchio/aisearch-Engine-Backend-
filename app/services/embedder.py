"""
Singleton embedding service using sentence-transformers.
Models are lazy-loaded on first use to prevent startup latency.
"""
import logging
from functools import lru_cache
from typing import List

from app.core.config import settings

logger = logging.getLogger(__name__)

_embedder = None


def get_embedder():
    """Return the singleton SentenceTransformer (lazy-init)."""
    global _embedder
    if _embedder is None:
        logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL)
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info("Embedding model loaded. Vector dim: %d", _embedder.get_sentence_embedding_dimension())
    return _embedder


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts. Returns list of float vectors."""
    model = get_embedder()
    vectors = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    return vectors.tolist()


def embed_query(query: str) -> List[float]:
    """Embed a single query string. Returns a flat float list."""
    return embed_texts([query])[0]


def get_vector_dim() -> int:
    """Return the embedding dimension for collection creation."""
    return get_embedder().get_sentence_embedding_dimension()
