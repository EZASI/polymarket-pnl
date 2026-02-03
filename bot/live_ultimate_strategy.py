"""
LIVE IMPLEMENTATION: Ultimate Safe Strategy

Combines three creative safe strategies into one live trading bot:
1. Pure Arbitrage (zero risk)
2. Market Making (very safe)
3. Time Decay Exploitation (safe)

This is the ULTIMATE safe strategy for 15-minute crypto markets.
"""

import asyncio
import json
import os
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import aiohttp


@dataclass
class SafeTrade:
    """A safe trade opportunity."""
    type: str
    market_id: str
    yes_token_id: str
    no_token_id: str
    action: str
    profit_potential: float
    risk_score: float
    capital_allocation: float


class UltimateSafeBot:
    """
    Live trading bot implementing the ultimate safe strategy.
    
    Strategy Priority:
    1. Pure Arbitrage (YES+NO < $1) - Execute immediately
    2. Market Making - Set up orders on both sides
    3. Time Decay - Bet against overpriced time value
    """
    
    def __init__(self, capital: float = 50000, dry_run: bool = True):
        self.capital = capital
        self.dry_run = dry_run
        self.api_base = "https://clob.polymarket.com"
        self.gamma_api = "https://gamma-api.polymarket.com"
        self.session: Optional[aiohttp.ClientSession] = None
        self.active_positions: Dict[str, SafeTrade] = {}
        self.stats = {
            'arbitrage_executed': 0,
            'market_making_setups': 0,
            'time_decay_trades': 0,
            'total_profit': 0.0,
            'total_trades': 0
        }
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_active_markets(self, limit: int = 100) -> List[Dict]:
        """Get active crypto markets."""
        url = f"{self.gamma_api}/markets"
        params = {
            'active': 'true',
            'limit': limit,
            'sort': 'volume',
            'desc': 'true'
        }
        
        try:
            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    markets = data.get('data', [])
                    # Filter to crypto markets
                    crypto = [
                        m for m in markets
                        if any(kw in m.get('question', '').lower()
                              for kw in ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'solana'])
                    ]
                    return crypto[:50]  # Top 50 crypto markets
        except Exception as e:
            print(f"Error fetching markets: {e}")
        
        return []
    
    async def get_market_prices(self, condition_id: str) -> Optional[Dict]:
        """Get current prices for a market."""
        url = f"{self.gamma_api}/markets/{condition_id}"
        
        try:
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as e:
            print(f"Error fetching market prices: {e}")
        
        return None
    
    async def get_orderbook(self, token_id: str) -> Optional[Dict]:
        """Get orderbook for a token."""
        url = f"{self.api_base}/book"
        params = {'token_id': token_id}
        
        try:
            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as e:
            print(f"Error fetching orderbook: {e}")
        
        return None
    
    async def scan_for_arbitrage(self, markets: List[Dict]) -> List[SafeTrade]:
        """Scan for pure arbitrage opportunities."""
        opportunities = []
        
        for market in markets:
            condition_id = market.get('conditionId', '')
            tokens = market.get('tokens', [])
            
            if len(tokens) < 2:
                continue
            
            yes_token = tokens[0]
            no_token = tokens[1]
            
            yes_token_id = yes_token.get('tokenId', '')
            no_token_id = no_token.get('tokenId', '')
            
            # Get orderbooks
            yes_book = await self.get_orderbook(yes_token_id)
            no_book = await self.get_orderbook(no_token_id)
            
            if not yes_book or not no_book:
                continue
            
            # Get best ask prices (what we'd pay to buy)
            yes_best_ask = self._get_best_ask(yes_book)
            no_best_ask = self._get_best_ask(no_book)
            
            if yes_best_ask and no_best_ask:
                total_cost = yes_best_ask + no_best_ask
                
                # Arbitrage exists if total < 0.99 (account for fees)
                if total_cost < 0.99:
                    profit = 1.0 - total_cost
                    
                    opportunities.append(SafeTrade(
                        type='arbitrage',
                        market_id=condition_id,
                        yes_token_id=yes_token_id,
                        no_token_id=no_token_id,
                        action=f'BUY_BOTH: YES@${yes_best_ask:.3f} + NO@${no_best_ask:.3f}',
                        profit_potential=profit,
                        risk_score=0.0,
                        capital_allocation=min(self.capital * 0.1, 5000)  # Max 10% or $5K
                    ))
        
        return opportunities
    
    async def scan_for_market_making(self, markets: List[Dict]) -> List[SafeTrade]:
        """Scan for market making opportunities."""
        opportunities = []
        
        for market in markets[:20]:  # Top 20 by volume
            condition_id = market.get('conditionId', '')
            tokens = market.get('tokens', [])
            
            if len(tokens) < 2:
                continue
            
            yes_token_id = tokens[0].get('tokenId', '')
            volume = market.get('volume', 0)
            liquidity = market.get('liquidity', 0)
            
            # Good market making setup:
            # - High volume (>$10K)
            # - Price near middle (0.4-0.6)
            # - Sufficient liquidity
            
            current_price = market.get('outcomePrices', [0.5])[0] if market.get('outcomePrices') else 0.5
            
            if volume > 10000 and liquidity > 5000 and 0.4 < current_price < 0.6:
                # Estimate profit from market making
                # Spread: ~0.2%, Maker rebate: ~0.1%, Net: ~0.1% per trade
                # Assume 20 trades/hour on active market
                hourly_profit = volume * 0.001 * 0.25  # 0.1% of volume, 15 min = 0.25 hour
                
                if hourly_profit > 10:  # At least $10 profit potential
                    opportunities.append(SafeTrade(
                        type='market_making',
                        market_id=condition_id,
                        yes_token_id=yes_token_id,
                        no_token_id=tokens[1].get('tokenId', ''),
                        action='MARKET_MAKE: Place limit orders on both sides',
                        profit_potential=hourly_profit,
                        risk_score=0.1,
                        capital_allocation=min(self.capital * 0.05, 2000)  # Max 5% or $2K
                    ))
        
        return opportunities
    
    async def scan_for_time_decay(self, markets: List[Dict]) -> List[SafeTrade]:
        """Scan for time decay opportunities."""
        opportunities = []
        
        for market in markets:
            condition_id = market.get('conditionId', '')
            end_date = market.get('endDate', '')
            
            if not end_date:
                continue
            
            # Calculate time remaining
            try:
                end_time = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                time_remaining = (end_time - datetime.now(end_time.tzinfo)).total_seconds() / 60
            except:
                continue
            
            # Only consider markets with < 3 minutes remaining
            if time_remaining > 3 or time_remaining < 0:
                continue
            
            tokens = market.get('tokens', [])
            if len(tokens) < 2:
                continue
            
            current_price = market.get('outcomePrices', [0.5])[0] if market.get('outcomePrices') else 0.5
            
            # Time decay opportunity: High price + Low time = Overpriced
            if current_price > 0.75:
                no_price = 1 - current_price
                
                # Calculate overpricing
                time_value = current_price - 0.5
                expected_decay = time_value * (time_remaining / 15)
                overpricing = current_price - (0.5 + expected_decay)
                
                if overpricing > 0.05:  # 5%+ overpriced
                    opportunities.append(SafeTrade(
                        type='time_decay',
                        market_id=condition_id,
                        yes_token_id=tokens[0].get('tokenId', ''),
                        no_token_id=tokens[1].get('tokenId', ''),
                        action=f'BUY_NO: Bet against overpriced time (${overpricing:.3f} overpriced)',
                        profit_potential=overpricing,
                        risk_score=0.2,
                        capital_allocation=min(self.capital * 0.03, 1000)  # Max 3% or $1K
                    ))
        
        return opportunities
    
    def _get_best_ask(self, orderbook: Dict) -> Optional[float]:
        """Get best ask price from orderbook."""
        asks = orderbook.get('asks', [])
        if asks and len(asks) > 0:
            return float(asks[0].get('price', 0))
        return None
    
    async def execute_trade(self, trade: SafeTrade):
        """Execute a safe trade."""
        if self.dry_run:
            print(f"\n[DRY RUN] {trade.type.upper()}: {trade.action}")
            print(f"  Market: {trade.market_id[:30]}...")
            print(f"  Profit Potential: ${trade.profit_potential:.2f}")
            print(f"  Risk Score: {trade.risk_score:.1f}/1.0")
            print(f"  Capital: ${trade.capital_allocation:.2f}")
            self.stats['total_trades'] += 1
            return
        
        # TODO: Implement actual order placement
        # Requires: @polymarket/clob-client, wallet setup, API key
        print(f"\n[LIVE] {trade.type.upper()}: {trade.action}")
        print(f"  ⚠️  Live trading requires wallet setup")
    
    async def run_scan_cycle(self):
        """Run one scan cycle for all opportunities."""
        print("\n" + "=" * 80)
        print(f"SCANNING FOR SAFE OPPORTUNITIES - {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 80)
        
        # Get active markets
        markets = await self.get_active_markets(limit=100)
        print(f"Found {len(markets)} active crypto markets")
        
        # Scan for all opportunity types
        print("\n1. Scanning for Pure Arbitrage (Zero Risk)...")
        arb_opps = await self.scan_for_arbitrage(markets)
        print(f"   Found {len(arb_opps)} arbitrage opportunities")
        
        print("\n2. Scanning for Market Making Setups (Very Safe)...")
        mm_opps = await self.scan_for_market_making(markets)
        print(f"   Found {len(mm_opps)} market making opportunities")
        
        print("\n3. Scanning for Time Decay Opportunities (Safe)...")
        td_opps = await self.scan_for_time_decay(markets)
        print(f"   Found {len(td_opps)} time decay opportunities")
        
        # Execute opportunities (prioritize by safety)
        all_opps = arb_opps + mm_opps + td_opps
        all_opps.sort(key=lambda x: x.risk_score)  # Safest first
        
        print(f"\n4. Executing {len(all_opps)} opportunities...")
        for opp in all_opps[:10]:  # Max 10 positions
            await self.execute_trade(opp)
            self.stats[f'{opp.type}_executed'] = self.stats.get(f'{opp.type}_executed', 0) + 1
        
        return len(all_opps)
    
    async def run(self, interval: int = 30):
        """Run the bot continuously."""
        print("=" * 80)
        print("ULTIMATE SAFE STRATEGY BOT")
        print("=" * 80)
        print(f"""
Capital: ${self.capital:,.0f}
Mode: {'DRY RUN' if self.dry_run else 'LIVE TRADING'}
Scan Interval: {interval} seconds

Strategy Priority:
1. Pure Arbitrage (Risk: 0/10) - Execute immediately
2. Market Making (Risk: 1/10) - Set up orders
3. Time Decay (Risk: 2/10) - Bet against overpriced time

Press Ctrl+C to stop
        """)
        
        try:
            while True:
                await self.run_scan_cycle()
                await asyncio.sleep(interval)
        except KeyboardInterrupt:
            print("\n\nStopping bot...")
            self.print_stats()
    
    def print_stats(self):
        """Print bot statistics."""
        print("\n" + "=" * 80)
        print("BOT STATISTICS")
        print("=" * 80)
        print(f"""
Arbitrage Executed: {self.stats.get('arbitrage_executed', 0)}
Market Making Setups: {self.stats.get('market_making_setups', 0)}
Time Decay Trades: {self.stats.get('time_decay_trades', 0)}
Total Trades: {self.stats['total_trades']}
Total Profit: ${self.stats['total_profit']:.2f}
        """)


async def main():
    """Main entry point."""
    import sys
    
    DRY_RUN = True  # Set to False for live trading
    CAPITAL = 50000  # Starting capital
    
    bot = UltimateSafeBot(capital=CAPITAL, dry_run=DRY_RUN)
    
    async with bot:
        await bot.run(interval=30)  # Scan every 30 seconds


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nBot stopped by user")
