"""
Generic Redis cache layer — cache-miss → DB fetch → store in Redis pattern.
Used across all layers for pervasive caching with configurable TTLs.
"""
import json
import hashlib
import logging
from typing import Optional, Any, Callable, Awaitable

from app.db.redis_client import redis_client
from app.core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core: Get-or-Fetch Pattern
# ---------------------------------------------------------------------------
async def cache_get_or_fetch(
    key: str,
    fetcher: Callable[[], Awaitable[Any]],
    ttl: int = None,
    serialize: bool = True,
) -> Any:
    """
    Redis cache-aside pattern:
    1. Check Redis for cached value
    2. On miss → call fetcher() to get from DB
    3. Store result in Redis with TTL
    4. Return result

    Args:
        key:       Redis key (should be namespaced, e.g., "ws:abc123")
        fetcher:   Async function that returns the data from DB
        ttl:       Time-to-live in seconds (defaults to REDIS_DEFAULT_TTL)
        serialize: If True, JSON-serialize/deserialize the value
    """
    ttl = ttl or settings.REDIS_DEFAULT_TTL

    # 1. Try cache
    cached = await cache_get(key)
    if cached is not None:
        logger.debug("Cache HIT: %s", key)
        if serialize:
            try:
                return json.loads(cached)
            except (json.JSONDecodeError, TypeError):
                return cached
        return cached

    # 2. Cache miss — fetch from DB
    logger.debug("Cache MISS: %s", key)
    result = await fetcher()

    # 3. Store in Redis
    if result is not None:
        value = json.dumps(result, default=str) if serialize else str(result)
        await cache_set(key, value, ttl)

    return result


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
async def cache_get(key: str) -> Optional[str]:
    """Get a value from Redis. Returns None on miss or error."""
    try:
        pool = redis_client.redis_pool
        if not pool:
            return None
        return await pool.get(key)
    except Exception:
        return None


async def cache_set(key: str, value: str, ttl: int = None) -> None:
    """Set a value in Redis with TTL."""
    try:
        pool = redis_client.redis_pool
        if not pool:
            return
        ttl = ttl or settings.REDIS_DEFAULT_TTL
        await pool.setex(key, ttl, value)
    except Exception:
        pass


async def cache_delete(key: str) -> None:
    """Delete a key from Redis (for cache invalidation)."""
    try:
        pool = redis_client.redis_pool
        if pool:
            await pool.delete(key)
    except Exception:
        pass


async def cache_delete_pattern(pattern: str) -> None:
    """Delete all keys matching a pattern (e.g., 'ws:abc123:*')."""
    try:
        pool = redis_client.redis_pool
        if not pool:
            return
        cursor = 0
        while True:
            cursor, keys = await pool.scan(cursor=cursor, match=pattern, count=100)
            if keys:
                await pool.delete(*keys)
            if cursor == 0:
                break
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Key builders (namespaced)
# ---------------------------------------------------------------------------
def user_key(user_id: str) -> str:
    return f"user:{user_id}"


def workspace_key(workspace_id: str) -> str:
    return f"ws:{workspace_id}"


def workspace_list_key(user_id: str) -> str:
    return f"ws_list:{user_id}"


def workspace_docs_key(workspace_id: str) -> str:
    return f"ws_docs:{workspace_id}"


def search_cache_key(query: str) -> str:
    h = hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]
    return f"search:{h}"


def rag_cache_key(query: str, workspace_id: str) -> str:
    h = hashlib.sha256(f"{query}:{workspace_id}".strip().lower().encode()).hexdigest()[:16]
    return f"rag:{h}"


def session_key(token: str) -> str:
    return f"session:{hashlib.sha256(token.encode()).hexdigest()[:16]}"
