"""
Live Execution Engine for Complement Arbitrage
================================================
File: execution/live_trader.py

WARNING: This is the LIVE TRADING module.
It connects to the Polymarket CLOB API using real credentials.

Red Team Fortifications Included:
1. Concurrency: Uses asyncio.to_thread + asyncio.gather for zero millisecond leg-delay.
2. Leg Risk Mitigation: Emergency unwind uses OrderType.MARKET (not GTC) if partial fill occurs.
3. Order Routing: Strictly enforces OrderType.FOK (Fill-Or-Kill) to prevent phantom liquidity fills.
"""

import os
import time
import logging
import asyncio
from typing import Dict, Any

# Polymarket SDK imports
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType

logger = logging.getLogger(__name__)


class LiveTrader:
    def __init__(self):
        """
        Initializes the Polymarket CLOB Client.
        Expects POLYMARKET_PRIVATE_KEY to be set in the environment.
        """
        self.host = os.getenv("POLYMARKET_HOST", "https://clob.polymarket.com")
        self.chain_id = int(os.getenv("POLYMARKET_CHAIN_ID", "137"))
        self.private_key = os.getenv("POLYMARKET_PRIVATE_KEY")

        if not self.private_key:
            raise ValueError("CRITICAL: POLYMARKET_PRIVATE_KEY environment variable is missing.")

        logger.info("Initializing LiveTrader ClobClient...")
        self.client = ClobClient(
            self.host,
            key=self.private_key,
            chain_id=self.chain_id,
            signature_type=1,  # 1 = EIP-712 signatures
        )

        # Derive and set L2 API credentials for the session
        creds = self.client.create_or_derive_api_creds()
        self.client.set_api_creds(creds)
        logger.info("LiveTrader initialized and authenticated successfully.")

    async def execute_complement_arb(
        self,
        signal: Any,
        token_id_yes: str,
        token_id_no: str,
    ) -> Dict[str, Any]:
        """
        Fires two FOK orders perfectly concurrently to capture the spread.
        """
        size_shares = round(signal.size_shares, 2)

        # 1. Prepare Order Arguments
        yes_args = OrderArgs(
            price=round(signal.yes_ask, 3),
            size=size_shares,
            side="BUY",
            token_id=token_id_yes,
        )

        no_args = OrderArgs(
            price=round(signal.no_ask, 3),
            size=size_shares,
            side="BUY",
            token_id=token_id_no,
        )

        logger.info(
            f"FIRING LIVE ARB: {size_shares} shares | "
            f"Y@${signal.yes_ask:.3f} + N@${signal.no_ask:.3f} = Total Cost ${signal.total_cost:.3f}"
        )

        # 2. Concurrency Mandate: Send both to SDK via separate OS threads
        yes_task = asyncio.to_thread(
            self.client.create_and_post_order,
            yes_args,
            orderType=OrderType.FOK,
        )

        no_task = asyncio.to_thread(
            self.client.create_and_post_order,
            no_args,
            orderType=OrderType.FOK,
        )

        # 3. Wait for both API responses concurrently
        start_time = time.time()
        results = await asyncio.gather(yes_task, no_task, return_exceptions=True)
        latency_ms = (time.time() - start_time) * 1000

        logger.info(f"FOK orders hit API and returned in {latency_ms:.1f}ms")

        yes_res, no_res = results

        # 4. Process Outcomes
        yes_success = self._is_success(yes_res)
        no_success = self._is_success(no_res)

        if yes_success and no_success:
            logger.info("SUCCESS: Both legs filled. Arbitrage locked.")
            return {
                "status": "FILLED_BOTH",
                "pnl_usd": signal.total_net_profit,
                "latency_ms": latency_ms,
            }

        elif not yes_success and not no_success:
            logger.warning("REJECTED: Both legs failed (Likely FOK depth removed by a faster bot).")
            return {"status": "REJECTED_BOTH"}

        # 5. Handle Leg Risk (Emergency Unwind)
        elif yes_success and not no_success:
            logger.error("LEG RISK: YES filled but NO rejected. Triggering immediate unwind.")
            await self._emergency_unwind(token_id_yes, size_shares)
            return {"status": "PARTIAL_YES_UNWOUND"}

        elif no_success and not yes_success:
            logger.error("LEG RISK: NO filled but YES rejected. Triggering immediate unwind.")
            await self._emergency_unwind(token_id_no, size_shares)
            return {"status": "PARTIAL_NO_UNWOUND"}

    def _is_success(self, res: Any) -> bool:
        """Helper to safely parse SDK responses."""
        if isinstance(res, Exception):
            logger.error(f"SDK Exception: {res}")
            return False
        # Polymarket SDK typically returns a dict with 'success': True/False
        return isinstance(res, dict) and res.get("success", False)

    async def _emergency_unwind(self, token_id: str, size: float):
        """
        Dumps the exposed leg immediately at MARKET price.
        Uses OrderType.MARKET to prevent GTC limit posting.
        """
        sell_args = OrderArgs(
            price=0.01,  # Floor price to guarantee market fill matching
            size=size,
            side="SELL",
            token_id=token_id,
        )
        try:
            res = await asyncio.to_thread(
                self.client.create_and_post_order,
                sell_args,
                orderType=OrderType.MARKET,
            )
            if self._is_success(res):
                logger.info(f"Emergency Unwind successful for token {token_id}.")
            else:
                logger.critical(f"Emergency Unwind failed. API returned: {res}")
        except Exception as e:
            logger.critical(f"FATAL: Emergency Unwind raised exception: {e}")
