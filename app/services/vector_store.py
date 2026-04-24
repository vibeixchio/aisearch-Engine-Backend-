"""
Qdrant vector store service.
Handles collection management, upsert, and hybrid (semantic + keyword) search.
"""
import logging
import asyncio
import uuid
from typing import List, Dict, Any, Optional

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    TextIndexParams,
    TokenizerType,
)

from app.core.config import settings
from app.services.embedder import get_vector_dim

logger = logging.getLogger(__name__)

_client: Optional[AsyncQdrantClient] = None


def get_qdrant_client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        kwargs = {"url": settings.QDRANT_URL}
        if settings.QDRANT_API_KEY:
            kwargs["api_key"] = settings.QDRANT_API_KEY
        _client = AsyncQdrantClient(**kwargs)
    return _client

async def close_qdrant_client():
    global _client
    if _client is not None:
        try:
            await _client.close()
            logger.info("Qdrant client closed")
        except Exception as e:
            logger.warning("Error closing Qdrant client: %s", e)
        _client = None


async def ensure_collection() -> None:
    """Create the collection if it does not already exist."""
    client = get_qdrant_client()
    exists = await client.collection_exists(settings.QDRANT_COLLECTION)
    if not exists:
        dim = get_vector_dim()
        await client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        # Create payload index for workspace_id filtering
        await client.create_payload_index(
            collection_name=settings.QDRANT_COLLECTION,
            field_name="workspace_id",
            field_schema="keyword",
        )
        # Create full-text index for BM25-style keyword search
        await client.create_payload_index(
            collection_name=settings.QDRANT_COLLECTION,
            field_name="text",
            field_schema=TextIndexParams(
                type="text",
                tokenizer=TokenizerType.WORD,
                min_token_len=2,
                max_token_len=20,
                lowercase=True,
            ),
        )
        logger.info(
            "Created Qdrant collection '%s' with dim=%d",
            settings.QDRANT_COLLECTION,
            dim,
        )
    else:
        logger.debug("Qdrant collection '%s' already exists", settings.QDRANT_COLLECTION)

    # Ensure semantic_cache collection exists
    semantic_cache_coll = "semantic_cache"
    exists = await client.collection_exists(semantic_cache_coll)
    if not exists:
        dim = get_vector_dim()
        await client.create_collection(
            collection_name=semantic_cache_coll,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection '%s' with dim=%d", semantic_cache_coll, dim)
    else:
        logger.debug("Qdrant collection '%s' already exists", semantic_cache_coll)


async def upsert_chunks(chunks: List[Dict[str, Any]]) -> int:
    """
    Upsert a list of chunk dicts into Qdrant.
    Each chunk must have 'vector' (list[float]) and 'payload' (dict).
    Returns the number of points upserted.
    """
    client = get_qdrant_client()
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=chunk["vector"],
            payload=chunk["payload"],
        )
        for chunk in chunks
    ]
    await client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points)
    logger.info("Upserted %d points to Qdrant", len(points))
    return len(points)


def _build_filter(workspace_id: Optional[str]) -> Optional[Filter]:
    if not workspace_id:
        return None
    return Filter(
        must=[FieldCondition(key="workspace_id", match=MatchValue(value=workspace_id))]
    )


async def vector_search(
    query_vector: List[float],
    workspace_id: Optional[str] = None,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Semantic vector search. Returns list of payload dicts with score."""
    client = get_qdrant_client()
    results = await client.search(
        collection_name=settings.QDRANT_COLLECTION,
        query_vector=query_vector,
        query_filter=_build_filter(workspace_id),
        limit=top_k,
        with_payload=True,
    )
    return [
        {**r.payload, "score": r.score, "search_type": "vector"}
        for r in results
    ]


async def keyword_search(
    query_text: str,
    workspace_id: Optional[str] = None,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Full-text keyword search using Qdrant's payload text index."""
    client = get_qdrant_client()

    must_conditions = []
    if workspace_id:
        must_conditions.append(
            FieldCondition(key="workspace_id", match=MatchValue(value=workspace_id))
        )
    # Full-text match condition
    must_conditions.append(
        models.FieldCondition(
            key="text",
            match=models.MatchText(text=query_text),
        )
    )

    results = await client.scroll(
        collection_name=settings.QDRANT_COLLECTION,
        scroll_filter=Filter(must=must_conditions),
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )
    return [
        {**point.payload, "score": 1.0, "search_type": "keyword"}
        for point in results[0]
    ]


async def delete_by_workspace(workspace_id: str) -> None:
    """Delete all points belonging to a workspace (useful for cleanup)."""
    client = get_qdrant_client()
    await client.delete(
        collection_name=settings.QDRANT_COLLECTION,
        points_selector=models.FilterSelector(filter=_build_filter(workspace_id)),
    )
    logger.info("Deleted all points for workspace_id='%s'", workspace_id)
