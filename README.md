# Savey LLM Service

Async LLM service for Savey money tracking app using FastStream, Redis PubSub, and LangChain.

## Architecture

- **FastStream**: Async message processing framework
- **Redis PubSub**: Message queue for async communication
- **LangChain**: LLM orchestration with OpenAI
- **Structured Outputs**: Pydantic-based validation
- **Function Calling**: Tool execution for transaction operations

## Message Flow

```
savey_api → Redis PubSub (llm:messages:input)
          → savey_llm (FastStream consumer)
          → LangChain + OpenAI
          → Tool Execution (if needed)
          → Redis PubSub (llm:messages:output)
          → savey_api
```

## Directory Structure

```
savey_llm/
├── core/           # Configuration and Redis setup
├── schemas/        # Pydantic schemas for messages and tools
├── services/       # LLM and tool services
├── routes/         # FastStream message handlers
├── tools/          # Function calling tools
├── app.py          # Application entry point
├── worker.py       # Worker entry point for container
└── Dockerfile      # Container definition
```

## Setup

1. **Install dependencies**:
```bash
uv sync
```

2. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

3. **Run locally**:
```bash
python worker.py
```

4. **Run with Docker Compose** (from savey_api directory):
```bash
cd ../savey_api
docker-compose up -d
```

## Features

### ✅ Async Message Processing
- FastStream-based Redis PubSub consumer
- Non-blocking message handling
- Automatic retry and error handling

### ✅ LangChain Integration
- OpenAI API integration
- Structured output support
- Function calling (tool binding)
- Conversation context management

### ✅ Tool System
- Get transactions with filters
- Create new transactions
- Get categories
- Get balance summary

### ✅ Type Safety
- Full Pydantic validation
- Type-safe schemas
- Structured LLM outputs

## Available Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_transactions` | Get user transactions | `transaction_type`, `start_date`, `end_date`, `category` |
| `create_transaction` | Create new transaction | `amount`, `category`, `description`, `transaction_type`, `date` |
| `get_categories` | Get all categories | None |
| `get_balance` | Get balance summary | None |

## Configuration

Environment variables:

- `REDIS_URL`: Redis connection URL
- `REDIS_CHANNEL_INPUT`: Input channel name
- `REDIS_CHANNEL_OUTPUT`: Output channel name
- `OPENAI_API_KEY`: OpenAI API key
- `OPENAI_MODEL`: Model name (default: gpt-4o-mini)
- `OPENAI_TEMPERATURE`: Temperature (default: 0.7)
- `MAX_TOKENS`: Max tokens per response
- `ENABLE_FUNCTION_CALLING`: Enable tool calling
- `SAVEY_API_URL`: Savey API base URL
- `LOG_LEVEL`: Logging level

## Development

### Adding New Tools

1. Define schema in `schemas/tools.py`:
```python
class NewTool(BaseModel):
    param1: str
    param2: int
```

2. Add tool definition in `services/tool_service.py`:
```python
{
    "name": "new_tool",
    "description": "Tool description",
    "parameters": NewTool.model_json_schema()
}
```

3. Implement execution in `services/tool_service.py`:
```python
async def execute_new_tool(self, user_id: str, **kwargs):
    # Implementation
    pass
```

### Testing

Test message publishing:
```python
import redis.asyncio as redis
import json

r = await redis.from_url("redis://localhost:6379")
await r.publish("llm:messages:input", json.dumps({
    "user_id": "test-user",
    "message_id": "test-msg",
    "content": "Show my transactions",
    "timestamp": "2024-01-15T10:00:00Z"
}))
```

## TODO

- [ ] Implement HTTP client for savey_api tool calls
- [ ] Add conversation memory/context
- [ ] Add retry logic for failed tool calls
- [ ] Add metrics and monitoring
- [ ] Add comprehensive tests
- [ ] Add streaming responses
- [ ] Add rate limiting
- [ ] Add caching for common queries

## License

MIT
