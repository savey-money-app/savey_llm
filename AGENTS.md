# Savey LLM Service

Async LLM worker that consumes chat messages from a Redis queue, processes them through PydanticAI agents, and publishes responses back via Redis PubSub. See `README.md` for architecture details.

## Cursor Cloud specific instructions

### Services

| Service | How to start | Notes |
|---------|-------------|-------|
| **Redis** | `redis-server --daemonize yes` | Required. Worker blocks on `brpop` and won't start without it. Verify with `redis-cli ping`. |
| **Worker** | `.venv/bin/python worker.py` | Main entry point. Listens on Redis queue `chat_queue`, publishes responses to `chat:<message_id>` channels. |

### Environment

- Copy `.env.example` to `.env` and adjust for local dev:
  - `REDIS_URL=redis://localhost:6379`
  - `SAVEY_API_URL=http://localhost:8000`
- **Gotcha**: `.env.example` contains `GEMINI_TEMPERATURE` but the Settings model expects `LLM_TEMPERATURE`. Rename or remove `GEMINI_TEMPERATURE` in `.env` to avoid a `ValidationError` (extra inputs not permitted).
- LLM API keys (`OPENAI_API_KEY` or `GEMINI_API_KEY`) are needed for full end-to-end LLM processing. Without them, the worker starts and processes messages but returns graceful error responses.

### Lint / Check

- No project-specific linter is configured. Use `ruff check .` (install with `uv pip install ruff`) for basic linting.
- Syntax check all files: `find . -name '*.py' -not -path './.venv/*' -exec python -m py_compile {} +`
- No test suite exists in the repo.

### Testing a message flow

Push a test job to Redis and subscribe for the response:
```bash
redis-cli LPUSH chat_queue '{"user_id":"123e4567-e89b-12d3-a456-426614174000","message_id":"test-001","content":"Hello","timestamp":"2024-01-15T10:30:00Z"}'
redis-cli SUBSCRIBE chat:test-001
```

### External dependencies not in this repo

- **savey_api**: REST backend (separate repo, expected at `SAVEY_API_URL`). Tool calls will fail without it, but the worker still operates.
- **Tesseract OCR / Poppler**: System packages for bank statement parsing. Not needed for basic chat flows.
