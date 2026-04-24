import logging
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

logger = logging.getLogger(__name__)


class MongoDB:
    client: AsyncIOMotorClient = None
    db = None


db = MongoDB()


async def connect_to_mongo():
    """Connect to MongoDB with connection validation."""
    try:
        db.client = AsyncIOMotorClient(
            settings.MONGODB_URL,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
        )
        db.db = db.client[settings.MONGODB_DB]
        # Validate connection
        await db.client.admin.command("ping")
        logger.info("Connected to MongoDB", extra={"db": settings.MONGODB_DB})
    except Exception as e:
        logger.error("Failed to connect to MongoDB", extra={"error": str(e)})
        raise


async def close_mongo_connection():
    if db.client:
        db.client.close()
        logger.info("Closed MongoDB connection")
