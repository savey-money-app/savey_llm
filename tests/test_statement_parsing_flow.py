import asyncio
from datetime import datetime
from uuid import uuid4

from schemas.api_tools import TransactionCreateShort
from schemas.bank_statement import ParsedStatement
from schemas.hitl import HITLFlowState, HITLFlowType
from services.hitl_flows.statement_parsing import StatementParsingFlow


class FakeHitlManager:
    def __init__(self):
        self.created = []
        self.deleted = []
        self.flow = None
        self.iteration = 1
        self.state_updates = []

    async def create_flow(self, **kwargs):
        self.created.append(kwargs)
        return type("FlowRequest", (), {"flow_id": "flow-1"})()

    async def get_flow(self, _user_id):
        return self.flow

    async def increment_iteration(self, _user_id):
        self.iteration += 1
        return self.iteration

    async def update_flow_state(self, user_id, state, data=None):
        self.state_updates.append((user_id, state, data))

    async def delete_flow(self, user_id):
        self.deleted.append(user_id)


class FakeAPIClient:
    def __init__(self):
        self.bulk_calls = []

    async def create_transactions_from_statement(self, **kwargs):
        self.bulk_calls.append(kwargs)
        return {"created_count": len(kwargs["transactions"]), "statement_id": "statement-1"}, None


def transaction(amount: float, category: str) -> TransactionCreateShort:
    return TransactionCreateShort(
        amount=amount,
        category=category,
        description=f"{category} transaction",
        date=datetime(2026, 5, 22, 10, 30),
    )


def test_initiate_parsing_flow_stores_currency_and_summary():
    async def run():
        hitl = FakeHitlManager()
        flow = StatementParsingFlow(hitl, FakeAPIClient())
        user_id = uuid4()
        parsed = ParsedStatement(transactions=[transaction(125, "Salary"), transaction(-15, "Food")])

        presentation = await flow.initiate_parsing_flow(
            user_id,
            "message-1",
            parsed,
            user_currency="KZT",
        )

        assert presentation.flow_id == "flow-1"
        assert presentation.total_income == 125
        assert presentation.total_expenses == 15
        assert "KZT" in presentation.message
        assert hitl.created[0]["flow_type"] == HITLFlowType.STATEMENT_PARSING
        assert hitl.created[0]["data"]["user_currency"] == "KZT"

    asyncio.run(run())


def test_modification_iteration_preserves_state_and_increments_counter():
    async def run():
        hitl = FakeHitlManager()
        statement_date = datetime(2026, 5, 1, 8, 0)
        hitl.flow = {
            "flow_id": "flow-2",
            "iteration": "1",
            "data": {
                "transactions": [transaction(-10, "Food").model_dump(mode="json")],
                "statement_date": statement_date.isoformat(),
                "iteration": 1,
                "user_remarks": None,
                "user_currency": "EUR",
            },
        }
        flow = StatementParsingFlow(hitl, FakeAPIClient())
        user_id = uuid4()
        modified = [transaction(-20, "Travel")]

        presentation = await flow.handle_modification_iteration(
            user_id,
            modified,
            remarks="taxi instead",
        )

        user_id_str, state, data = hitl.state_updates[0]
        assert user_id_str == str(user_id)
        assert state == HITLFlowState.IN_PROGRESS
        assert data["iteration"] == 2
        assert data["user_currency"] == "EUR"
        assert data["statement_date"] == statement_date.isoformat()
        assert data["user_remarks"] == "taxi instead"
        assert presentation.flow_id == "flow-2"
        assert "EUR" in presentation.message
        assert "Iteration 2" in presentation.message

    asyncio.run(run())


def test_execute_bulk_creation_uses_updated_flow_and_deletes_it():
    async def run():
        hitl = FakeHitlManager()
        api_client = FakeAPIClient()
        user_id = uuid4()
        statement_date = datetime(2026, 5, 10, 9, 15)
        stored_transaction = transaction(-12, "Food")
        hitl.flow = {
            "flow_id": "flow-3",
            "data": {
                "transactions": [stored_transaction.model_dump(mode="json")],
                "statement_date": statement_date.isoformat(),
                "iteration": 2,
                "user_remarks": "updated",
                "user_currency": "GBP",
            },
        }
        flow = StatementParsingFlow(hitl, api_client)

        message, balance = await flow.execute_bulk_creation(user_id)

        assert balance is None
        assert "Created **1 transactions**" in message
        assert api_client.bulk_calls[0]["statement_date"] == statement_date
        assert api_client.bulk_calls[0]["transactions"][0] == stored_transaction
        assert hitl.deleted == [str(user_id)]

    asyncio.run(run())
