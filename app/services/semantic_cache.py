import logging
import json
import uuid
import asyncio
from typing import Optional, Dict, Any, Callable, Awaitable

from qdrant_client.http.models import PointStruct

from app.services.vector_store import get_qdrant_client
from app.services.embedder import embed_query
from app.core.config import settings

logger = logging.getLogger(__name__)

SEMANTIC_CACHE_COLLECTION = "semantic_cache"
SIMILARITY_THRESHOLD = 0.95  # Cosine similarity threshold


async def get_semantic_cache(query: str) -> Optional[Any]:
    """Check if a semantically similar query exists in the cache."""
    try:
        query_vector = await asyncio.to_thread(embed_query, query)
        client = get_qdrant_client()

        results = await client.search(
            collection_name=SEMANTIC_CACHE_COLLECTION,
            query_vector=query_vector,
            limit=1,
            with_payload=True,
            score_threshold=SIMILARITY_THRESHOLD,
        )

        if results:
            logger.info("Semantic cache HIT for query: '%s' (matched '%s' with score %.3f)", 
                        query, results[0].payload.get("query"), results[0].score)
            
            # The cached result is stored as a JSON string in the payload
            cached_result_str = results[0].payload.get("result")
            if cached_result_str:
                return json.loads(cached_result_str)
                
    except Exception as e:
        logger.warning("Semantic cache fetch failed: %s", e)
        
    logger.debug("Semantic cache MISS for query: '%s'", query)
    return None


async def set_semantic_cache(query: str, result: Any) -> None:
    """Store the query and result in the semantic cache."""
    try:
        query_vector = await asyncio.to_thread(embed_query, query)
        client = get_qdrant_client()
        
        result_str = json.dumps(result, default=str)
        
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=query_vector,
            payload={
                "query": query,
                "result": result_str,
            }
        )
        
        await client.upsert(
            collection_name=SEMANTIC_CACHE_COLLECTION,
            points=[point]
        )
        logger.debug("Saved to semantic cache: '%s'", query)
    except Exception as e:
        logger.warning("Failed to save to semantic cache: %s", e)


async def semantic_cache_get_or_fetch(
    query: str,
    fetcher: Callable[[], Awaitable[Any]],
) -> Any:
    """
    Semantic Cache Pattern:
    1. Check Qdrant for queries with >0.95 cosine similarity.
    2. If found, return cached result.
    3. If miss, run fetcher() and store the new result in Qdrant.
    """
    cached = await get_semantic_cache(query)
    if cached is not None:
        return cached

    result = await fetcher()

    if result is not None:
        # Fire and forget the cache storage to not block the response
        asyncio.create_task(set_semantic_cache(query, result))

    return result
