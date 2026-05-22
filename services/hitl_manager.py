"""
HITL (Human-in-the-Loop) Manager

Manages human confirmation flows with Redis state storage.
Handles flow creation, state management, and user responses.

Flows are keyed by user_id (one active flow per user at a time).
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

import redis.asyncio as redis
from core.config import settings
from schemas.hitl import (
    HITLFlowState,
    HITLFlowType,
    HITLRequest,
)

logger = logging.getLogger(__name__)


class HITLManager:
    """Manager for Human-in-the-Loop confirmation flows"""

    def __init__(self):
        self.redis_url = settings.REDIS_URL
        self.flow_prefix = settings.REDIS_HITL_PREFIX
        self.flow_ttl = settings.HITL_FLOW_TTL
        self.max_iterations = settings.HITL_MAX_ITERATIONS
        self._redis: Any = None

    async def _get_redis(self) -> Any:
        """Get or create Redis connection"""
        if self._redis is None:
            self._redis = await redis.from_url(self.redis_url, decode_responses=True)
        return self._redis

    def _flow_key(self, user_id: str) -> str:
        """Generate Redis key for a user's active flow"""
        return f"{self.flow_prefix}user:{user_id}"

    async def create_flow(
        self, user_id: UUID, message_id: str, flow_type: HITLFlowType, data: Dict[str, Any]
    ) -> HITLRequest:
        """
        Create a new HITL flow for a user.

        Any existing active flow for this user is automatically cancelled/deleted
        before the new flow is created.

        Args:
            user_id: User ID
            message_id: Original message ID
            flow_type: Type of HITL flow
            data: Flow-specific data

        Returns:
            HITLRequest with flow ID and details
        """
        flow_id = str(uuid4())
        expires_at = datetime.utcnow() + timedelta(seconds=self.flow_ttl)
        user_id_str = str(user_id)

        # Auto-cancel any existing flow for this user
        existing = await self.get_flow(user_id_str)
        if existing:
            logger.info(
                f"🔄 Auto-cancelling stale flow {existing.get('flow_id')} for user {user_id_str}"
            )
            await self.delete_flow(user_id_str)

        request = HITLRequest(
            flow_id=flow_id,
            user_id=user_id,
            message_id=message_id,
            flow_type=flow_type,
            data=data,
            expires_at=expires_at,
        )

        # Store in Redis keyed by user_id
        r = await self._get_redis()
        flow_data = {
            "flow_id": flow_id,
            "user_id": user_id_str,
            "message_id": message_id,
            "flow_type": flow_type.value,
            "state": HITLFlowState.PENDING.value,
            "data": json.dumps(data),
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": expires_at.isoformat(),
            "iteration": 1,
        }

        key = self._flow_key(user_id_str)
        await r.hset(key, mapping=flow_data)
        await r.expire(key, self.flow_ttl)

        logger.info(f"📝 Created HITL flow {flow_id} for user {user_id} (type: {flow_type.value})")
        return request

    async def get_flow(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get flow data from Redis by user_id.

        Args:
            user_id: User ID string

        Returns:
            Flow data dict or None if not found
        """
        r = await self._get_redis()
        flow_data = await r.hgetall(self._flow_key(user_id))

        if not flow_data:
            return None

        # Parse JSON data field
        flow_data["data"] = json.loads(flow_data["data"])
        return flow_data

    async def get_active_flow(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get active flow for a user (PENDING or IN_PROGRESS state only).

        Args:
            user_id: User ID string

        Returns:
            Flow data dict if an active flow exists, else None
        """
        flow = await self.get_flow(user_id)
        if not flow:
            return None

        state = HITLFlowState(flow["state"])
        if state in (HITLFlowState.PENDING, HITLFlowState.IN_PROGRESS):
            return flow

        return None

    async def update_flow_state(
        self, user_id: str, state: HITLFlowState, data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update flow state.

        Args:
            user_id: User ID string
            state: New state
            data: Optional updated data
        """
        r = await self._get_redis()
        updates = {"state": state.value}

        if data is not None:
            updates["data"] = json.dumps(data)

        key = self._flow_key(user_id)
        await r.hset(key, mapping=updates)
        logger.info(f"🔄 Updated flow for user {user_id} state to {state.value}")

    async def increment_iteration(self, user_id: str) -> int:
        """
        Increment flow iteration counter.

        Args:
            user_id: User ID string

        Returns:
            New iteration number
        """
        r = await self._get_redis()
        new_iteration = await r.hincrby(self._flow_key(user_id), "iteration", 1)
        logger.info(f"🔢 Flow for user {user_id} iteration incremented to {new_iteration}")
        return new_iteration

    async def delete_flow(self, user_id: str) -> None:
        """
        Delete a flow from Redis.

        Args:
            user_id: User ID string
        """
        r = await self._get_redis()
        await r.delete(self._flow_key(user_id))
        logger.info(f"🗑️ Deleted flow for user {user_id}")

    async def close(self):
        """Close Redis connection"""
        if self._redis:
            await self._redis.close()
