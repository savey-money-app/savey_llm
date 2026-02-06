"""Redis connection utilities"""
from redis.asyncio import Redis
from core.config import settings
import logging

logger = logging.getLogger(__name__)


async def get_redis() -> Redis:
    """Get Redis connection"""
    try:
        redis = await Redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
        logger.info(f"Connected to Redis at {settings.REDIS_URL}")
        return redis
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        raise
