"""
Search result cache — wraps the cache_layer with search-specific logic.
Configurable TTL from settings (5-15 minutes).
"""
import logging
from typing import Optional

from app.services.cache_layer import cache_get, cache_set, search_cache_key
from app.core.config import settings

logger = logging.getLogger(__name__)


async def get_cached_response(query: str) -> Optional[str]:
    """Check Redis for a cached search result."""
    cached = await cache_get(search_cache_key(query))
    if cached:
        logger.info("Cache HIT", extra={"query": query[:60]})
    return cached


async def set_cached_response(query: str, response: str, ttl: int = None):
    """Store a search result in Redis with TTL."""
    ttl = ttl or settings.REDIS_SEARCH_TTL
    await cache_set(search_cache_key(query), response, ttl)
