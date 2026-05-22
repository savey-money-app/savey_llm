import asyncio
import json
from datetime import datetime, timedelta
from uuid import uuid4

from schemas.hitl import HITLFlowState, HITLFlowType
from services import hitl_manager


class FakeRedis:
    def __init__(self):
        self.hashes = {}
        self.expirations = {}
        self.deleted = []

    async def hset(self, key, mapping):
        self.hashes.setdefault(key, {}).update(mapping)

    async def hgetall(self, key):
        return self.hashes.get(key, {}).copy()

    async def expire(self, key, ttl):
        self.expirations[key] = ttl

    async def hincrby(self, key, field, amount):
        value = int(self.hashes[key].get(field, 0)) + amount
        self.hashes[key][field] = value
        return value

    async def delete(self, key):
        self.deleted.append(key)
        self.hashes.pop(key, None)

    async def close(self):
        return None


def test_create_flow_serializes_data_and_ttl(monkeypatch):
    async def run():
        redis = FakeRedis()
        manager = hitl_manager.HITLManager()
        manager.flow_ttl = 90
        manager._redis = redis
        created_at = datetime(2026, 5, 22, 10, 30)
        monkeypatch.setattr(hitl_manager, "utc_now", lambda: created_at)

        user_id = uuid4()
        request = await manager.create_flow(
            user_id,
            "message-1",
            HITLFlowType.TRANSACTION_DELETION,
            {"ids": ["transaction-1"]},
        )

        key = manager._flow_key(str(user_id))
        stored = redis.hashes[key]

        assert request.expires_at == created_at + timedelta(seconds=90)
        assert redis.expirations[key] == 90
        assert stored["flow_id"] == request.flow_id
        assert stored["flow_type"] == HITLFlowType.TRANSACTION_DELETION.value
        assert stored["state"] == HITLFlowState.PENDING.value
        assert stored["created_at"] == created_at.isoformat()
        assert json.loads(stored["data"]) == {"ids": ["transaction-1"]}

    asyncio.run(run())


def test_active_flow_parses_data_and_filters_terminal_states():
    async def run():
        redis = FakeRedis()
        manager = hitl_manager.HITLManager()
        manager._redis = redis
        user_id = "user-1"
        key = manager._flow_key(user_id)
        redis.hashes[key] = {
            "flow_id": "flow-1",
            "state": HITLFlowState.PENDING.value,
            "data": json.dumps({"iteration": 1}),
        }

        active = await manager.get_active_flow(user_id)
        assert active is not None
        assert active["data"] == {"iteration": 1}

        await manager.update_flow_state(user_id, HITLFlowState.CANCELLED)
        assert await manager.get_active_flow(user_id) is None

    asyncio.run(run())


def test_create_flow_replaces_existing_user_flow():
    async def run():
        redis = FakeRedis()
        manager = hitl_manager.HITLManager()
        manager._redis = redis
        user_id = uuid4()
        key = manager._flow_key(str(user_id))
        redis.hashes[key] = {
            "flow_id": "stale-flow",
            "state": HITLFlowState.PENDING.value,
            "data": json.dumps({"old": True}),
        }

        await manager.create_flow(
            user_id,
            "message-2",
            HITLFlowType.STATEMENT_PARSING,
            {"transactions": []},
        )

        assert redis.deleted == [key]
        assert redis.hashes[key]["message_id"] == "message-2"
        assert await manager.increment_iteration(str(user_id)) == 2

    asyncio.run(run())
