"""
HITL (Human-in-the-Loop) Manager

Manages human confirmation flows with Redis state storage.
Handles flow creation, state management, and user responses.
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
    HITLResponse,
    HITLUserResponse,
)

logger = logging.getLogger(__name__)


class HITLManager:
    """Manager for Human-in-the-Loop confirmation flows"""

    def __init__(self):
        self.redis_url = settings.REDIS_URL
        self.flow_prefix = settings.REDIS_HITL_PREFIX
        self.flow_ttl = settings.HITL_FLOW_TTL
        self.max_iterations = settings.HITL_MAX_ITERATIONS
        self._redis: Optional[redis.Redis] = None

    async def _get_redis(self) -> redis.Redis:
        """Get or create Redis connection"""
        if self._redis is None:
            self._redis = await redis.from_url(self.redis_url, decode_responses=True)
        return self._redis

    def _flow_key(self, flow_id: str) -> str:
        """Generate Redis key for flow"""
        return f"{self.flow_prefix}{flow_id}"

    async def create_flow(
        self, user_id: UUID, message_id: str, flow_type: HITLFlowType, data: Dict[str, Any]
    ) -> HITLRequest:
        """
        Create a new HITL flow

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

        request = HITLRequest(
            flow_id=flow_id,
            user_id=user_id,
            message_id=message_id,
            flow_type=flow_type,
            data=data,
            expires_at=expires_at,
        )

        # Store in Redis
        r = await self._get_redis()
        flow_data = {
            "flow_id": flow_id,
            "user_id": str(user_id),
            "message_id": message_id,
            "flow_type": flow_type.value,
            "state": HITLFlowState.PENDING.value,
            "data": json.dumps(data),
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": expires_at.isoformat(),
            "iteration": 1,
        }

        await r.hset(self._flow_key(flow_id), mapping=flow_data)
        await r.expire(self._flow_key(flow_id), self.flow_ttl)

        logger.info(f"📝 Created HITL flow {flow_id} for user {user_id} (type: {flow_type.value})")
        return request

    async def get_flow(self, flow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get flow data from Redis

        Args:
            flow_id: Flow ID

        Returns:
            Flow data dict or None if not found
        """
        r = await self._get_redis()
        flow_data = await r.hgetall(self._flow_key(flow_id))

        if not flow_data:
            return None

        # Parse JSON data field
        flow_data["data"] = json.loads(flow_data["data"])
        return flow_data

    async def update_flow_state(
        self, flow_id: str, state: HITLFlowState, data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update flow state

        Args:
            flow_id: Flow ID
            state: New state
            data: Optional updated data
        """
        r = await self._get_redis()
        updates = {"state": state.value}

        if data is not None:
            updates["data"] = json.dumps(data)

        await r.hset(self._flow_key(flow_id), mapping=updates)
        logger.info(f"🔄 Updated flow {flow_id} state to {state.value}")

    async def increment_iteration(self, flow_id: str) -> int:
        """
        Increment flow iteration counter

        Args:
            flow_id: Flow ID

        Returns:
            New iteration number
        """
        r = await self._get_redis()
        new_iteration = await r.hincrby(self._flow_key(flow_id), "iteration", 1)
        logger.info(f"🔢 Flow {flow_id} iteration incremented to {new_iteration}")
        return new_iteration

    async def process_user_response(
        self, flow_id: str, user_response: HITLUserResponse
    ) -> HITLResponse:
        """
        Process user's response to a HITL flow

        Args:
            flow_id: Flow ID
            user_response: User's response data

        Returns:
            HITLResponse with updated state
        """
        flow = await self.get_flow(flow_id)

        if not flow:
            return HITLResponse(
                flow_id=flow_id,
                flow_type=HITLFlowType.TRANSACTION_DELETION,  # Default
                state=HITLFlowState.EXPIRED,
                message="This confirmation flow has expired. Please start over.",
                requires_user_action=False,
            )

        flow_type = HITLFlowType(flow["flow_type"])
        current_state = HITLFlowState(flow["state"])

        # Check if flow already completed/cancelled
        if current_state in [HITLFlowState.CONFIRMED, HITLFlowState.CANCELLED, HITLFlowState.EXPIRED]:
            return HITLResponse(
                flow_id=flow_id,
                flow_type=flow_type,
                state=current_state,
                message=f"This flow has already been {current_state.value}.",
                requires_user_action=False,
            )

        # Process based on action
        if user_response.action == "confirm":
            await self.update_flow_state(flow_id, HITLFlowState.CONFIRMED)
            return HITLResponse(
                flow_id=flow_id,
                flow_type=flow_type,
                state=HITLFlowState.CONFIRMED,
                message="✅ Confirmed! Processing your request...",
                data=flow["data"],
                requires_user_action=False,
            )

        elif user_response.action == "cancel":
            await self.update_flow_state(flow_id, HITLFlowState.CANCELLED)
            return HITLResponse(
                flow_id=flow_id,
                flow_type=flow_type,
                state=HITLFlowState.CANCELLED,
                message="❌ Cancelled. No changes were made.",
                requires_user_action=False,
            )

        elif user_response.action == "modify":
            # Check iteration limit
            current_iteration = int(flow.get("iteration", 1))
            if current_iteration >= self.max_iterations:
                await self.update_flow_state(flow_id, HITLFlowState.EXPIRED)
                return HITLResponse(
                    flow_id=flow_id,
                    flow_type=flow_type,
                    state=HITLFlowState.EXPIRED,
                    message=f"Maximum modification iterations ({self.max_iterations}) reached. Please start over.",
                    requires_user_action=False,
                )

            # Increment iteration and update data
            await self.increment_iteration(flow_id)

            # Merge modifications into flow data
            updated_data = flow["data"].copy()
            if user_response.modifications:
                updated_data.update(user_response.modifications)
            if user_response.comment:
                updated_data["user_remarks"] = user_response.comment

            await self.update_flow_state(flow_id, HITLFlowState.IN_PROGRESS, updated_data)

            return HITLResponse(
                flow_id=flow_id,
                flow_type=flow_type,
                state=HITLFlowState.IN_PROGRESS,
                message="🔄 Processing your modifications...",
                data=updated_data,
                requires_user_action=True,  # Will need another confirmation
            )

        else:
            return HITLResponse(
                flow_id=flow_id,
                flow_type=flow_type,
                state=current_state,
                message=f"Unknown action: {user_response.action}",
                requires_user_action=True,
            )

    async def delete_flow(self, flow_id: str) -> None:
        """
        Delete a flow from Redis

        Args:
            flow_id: Flow ID
        """
        r = await self._get_redis()
        await r.delete(self._flow_key(flow_id))
        logger.info(f"🗑️ Deleted flow {flow_id}")

    async def close(self):
        """Close Redis connection"""
        if self._redis:
            await self._redis.close()
