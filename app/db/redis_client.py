import logging
import redis.asyncio as redis
from app.core.config import settings

logger = logging.getLogger(__name__)


class RedisClient:
    redis_pool = None


redis_client = RedisClient()


async def connect_to_redis():
    """Connect to Redis with connection validation."""
    try:
        redis_client.redis_pool = redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        # Validate connection
        await redis_client.redis_pool.ping()
        logger.info("Connected to Redis")
    except Exception as e:
        logger.error("Failed to connect to Redis", extra={"error": str(e)})
        raise


async def close_redis_connection():
    if redis_client.redis_pool:
        await redis_client.redis_pool.close()
        logger.info("Closed Redis connection")
