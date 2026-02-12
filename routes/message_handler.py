"""FastStream message handler for Redis PubSub"""
from faststream import FastStream
from faststream.redis import RedisBroker
from core.config import settings
from schemas.message import MessageInput
from schemas.message import MessageOutput
from services.llm_service import LLMService
from services.model_factory import get_model_name
import logging
import json

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize broker and app
broker = RedisBroker(settings.REDIS_URL)
app = FastStream(broker)

# Initialize services
llm_service = LLMService()


@app.on_startup
async def on_startup():
    """Startup event handler"""
    logger.info("Savey LLM Service starting up...")
    logger.info(f"Redis URL: {settings.REDIS_URL}")
    logger.info(f"Input channel: {settings.REDIS_CHANNEL_INPUT}")
    logger.info(f"Output channel: {settings.REDIS_CHANNEL_OUTPUT}")
    logger.info(f"Provider: {settings.LLM_PROVIDER}, Model: {get_model_name('main')}")


@app.on_shutdown
async def on_shutdown():
    """Shutdown event handler"""
    logger.info("Savey LLM Service shutting down...")


@broker.subscriber(settings.REDIS_CHANNEL_INPUT)
@broker.publisher(settings.REDIS_CHANNEL_OUTPUT)
async def process_llm_message(message: MessageInput) -> MessageOutput:
    """
    Subscribe to input channel, process with LLM, publish to output channel

    This is the main message handler that:
    1. Receives messages from Redis PubSub input channel
    2. Processes them with LangChain/OpenAI
    3. Executes any tool calls if needed
    4. Publishes results to Redis PubSub output channel

    Args:
        message: MessageInput from Redis

    Returns:
        MessageOutput to be published to output channel
    """
    logger.info(f"📨 Received message {message.message_id} from user {message.user_id}")
    logger.debug(f"Message content: {message.content}")

    try:
        # Process message with LLM service
        response = await llm_service.process_message(message)

        logger.info(f"✅ Completed processing message {message.message_id}")
        logger.debug(f"Response: {response.content[:100]}...")

        if response.tool_calls:
            logger.info(f"🔧 Executed {len(response.tool_calls)} tool calls")

        return MessageOutput(
            content=response.content,
            hitl_data=response.hitl_data,
            balance=response.balance,
            error=response.error
        )

    except Exception as e:
        logger.error(f"❌ Error processing message {message.message_id}: {e}", exc_info=True)

        # Return error response
        return MessageOutput(
            content="I apologize, but I encountered an error processing your request. Please try again.",
            hitl_data=None,
            balance=None,
            error=str(e)
        )


# Health check subscriber (optional)
@broker.subscriber("llm:health:check")
async def health_check(message: dict):
    """Health check endpoint"""
    logger.info("Health check received")
    return {"status": "healthy", "service": "savey_llm"}
