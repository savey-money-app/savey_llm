"""Worker entry point — pops jobs from Redis queue, processes with LLM, publishes response"""
import asyncio
import json
import logging
import signal
import sys
from typing import Any

import redis.asyncio as aioredis

from core.config import settings
from schemas.message import MessageInput
from services.llm_service import LLMService
from services.model_factory import get_model_name

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

QUEUE_KEY = "chat_queue"
CHANNEL_PREFIX = "chat"


async def process_jobs(redis: Any, llm_service: LLMService) -> None:
    """Main loop: brpop a job, process it, publish response chunks."""
    logger.info(f"Listening on queue '{QUEUE_KEY}'...")
    while True:
        try:
            # Block until a job arrives (0 = no timeout)
            result = await redis.brpop(QUEUE_KEY, timeout=0)
            if result is None:
                continue

            _, raw = result
            job = json.loads(raw)
            message_id = job.get("message_id", "unknown")
            channel = f"{CHANNEL_PREFIX}:{message_id}"

            logger.info(f"📨 Processing job {message_id}")

            try:
                message = MessageInput(**job)

                tokens_published = 0

                async def on_token(token: str) -> None:
                    nonlocal tokens_published
                    tokens_published += 1
                    chunk = {"content": token, "hitl_data": None, "error": None}
                    await redis.publish(channel, json.dumps(chunk))

                response = await llm_service.stream_message(message, on_token)

                logger.info(f"✅ LLM processing complete for {message_id}, tokens_published={tokens_published}")

                if tokens_published > 0:
                    # Tokens were streamed — only publish a trailing chunk if there
                    # is metadata (balance, hitl_data) that wasn't part of the stream.
                    if response.balance or response.hitl_data or response.error:
                        meta: dict = {
                            "content": "",
                            "hitl_data": response.hitl_data,
                            "error": response.error,
                        }
                        if response.balance:
                            meta["balance"] = response.balance.model_dump()
                        await redis.publish(channel, json.dumps(meta))
                        logger.info(f"📤 Metadata chunk published for {message_id}")
                    else:
                        logger.info(f"📤 Stream complete for {message_id} (no extra metadata)")
                else:
                    # No tokens streamed (HITL / statement parser / error) —
                    # publish the complete response as a single chunk.
                    payload: dict = {
                        "content": response.content or "",
                        "hitl_data": response.hitl_data,
                        "error": response.error,
                    }
                    if response.balance:
                        payload["balance"] = response.balance.model_dump()
                    await redis.publish(channel, json.dumps(payload))
                    logger.info(f"📤 Complete response published for {message_id}")

            except Exception as e:
                logger.error(f"❌ Error processing job {message_id}: {e}", exc_info=True)
                await redis.publish(channel, json.dumps({"error": str(e), "content": None}))

            finally:
                # Always send [DONE] so the API-side stream closes
                await redis.publish(channel, "[DONE]")
                logger.info(f"🏁 Done signal sent for {message_id}")

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Queue error: {e}", exc_info=True)
            await asyncio.sleep(1)  # brief back-off before retrying


async def main():
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    logger.info("🚀 Starting Savey LLM Worker...")
    logger.info(f"Redis: {settings.REDIS_URL}")
    logger.info(f"Provider: {settings.LLM_PROVIDER}, Model: {get_model_name('main')}")

    redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    llm_service = LLMService()

    try:
        await process_jobs(redis, llm_service)
    finally:
        await redis.aclose()
        logger.info("Worker stopped.")


if __name__ == "__main__":
    asyncio.run(main())
