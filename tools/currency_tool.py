"""Currency conversion tool using real-time exchange rates"""

import logging
from typing import Any, Dict
from uuid import UUID

import httpx
from pydantic import BaseModel, Field

from tools.base import BaseTool

logger = logging.getLogger(__name__)

# Free exchange-rate API (no key required, ~1500 req/month)
_EXCHANGE_RATE_API_URL = "https://open.er-api.com/v6/latest/{base}"


# ============================================================================
# Parameter Schema
# ============================================================================


class CurrencyConverterParams(BaseModel):
    """Parameters for currency conversion"""

    amount: float = Field(..., description="The monetary amount to convert")
    from_currency: str = Field(
        ..., description="Source currency ISO 4217 code (e.g. 'USD', 'EUR', 'RUB')"
    )
    to_currency: str = Field(
        ..., description="Target currency ISO 4217 code (e.g. 'USD', 'EUR', 'RUB')"
    )


# ============================================================================
# Tool Implementation
# ============================================================================


class CurrencyConverterTool(BaseTool):
    """Convert an amount from one currency to another using real-time exchange rates"""

    name = "convert_currency"
    description = (
        "Convert a monetary amount from one currency to another using real-time "
        "exchange rates. Use this when the user mentions an amount in a currency "
        "different from their preferred currency."
    )
    args_schema = CurrencyConverterParams

    async def execute(self, user_id: UUID, arguments: Dict[str, Any]) -> Dict[str, Any]:
        amount = arguments["amount"]
        from_currency = arguments["from_currency"].upper().strip()
        to_currency = arguments["to_currency"].upper().strip()

        if from_currency == to_currency:
            return {
                "converted_amount": round(amount, 2),
                "rate": 1.0,
                "from_currency": from_currency,
                "to_currency": to_currency,
            }

        url = _EXCHANGE_RATE_API_URL.format(base=from_currency)

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()

            if data.get("result") != "success":
                raise ValueError(f"Exchange rate API error: {data.get('error-type', 'unknown')}")

            rates = data.get("rates", {})
            rate = rates.get(to_currency)

            if rate is None:
                raise ValueError(
                    f"Currency '{to_currency}' not found in exchange rate data"
                )

            converted = round(amount * rate, 2)

            logger.info(
                f"Converted {amount} {from_currency} -> {converted} {to_currency} (rate: {rate})"
            )

            return {
                "converted_amount": converted,
                "rate": rate,
                "from_currency": from_currency,
                "to_currency": to_currency,
            }

        except httpx.HTTPError as e:
            logger.error(f"Exchange rate API request failed: {e}")
            return {
                "error": f"Failed to fetch exchange rates: {e}",
                "success": False,
            }
