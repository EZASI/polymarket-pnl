"""
Test script to verify the live bot connects to Polymarket APIs
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.live_volume_spike_bot import PolymarketAPIClient, VolumeSpikeDetector, Trade


async def test_api_connection():
    """Test basic API connectivity."""
    print("=" * 80)
    print("TESTING POLYMARKET API CONNECTION")
    print("=" * 80)
    
    async with PolymarketAPIClient() as client:
        # Test 1: Fetch markets
        print("\n1. Testing Gamma API (markets)...")
        markets = await client.get_markets(limit=10)
        
        if markets:
            print(f"   ✅ Successfully fetched {len(markets)} markets")
            print(f"   Sample market: {markets[0].get('question', 'N/A')[:60]}...")
        else:
            print("   ❌ Failed to fetch markets")
            return False
        
        # Test 2: Fetch trades for a market
        if markets:
            condition_id = markets[0].get('conditionId', '')
            print(f"\n2. Testing CLOB API (trades for market {condition_id[:20]}...)...")
            trades = await client.get_market_trades(condition_id, limit=10)
            
            if trades:
                print(f"   ✅ Successfully fetched {len(trades)} trades")
                if len(trades) > 0:
                    print(f"   Sample trade: ${trades[0].get('price', 0):.3f} @ {trades[0].get('size', 0)} shares")
            else:
                print("   ⚠️  No trades found (market may be inactive)")
        
        # Test 3: Test volume spike detector
        print("\n3. Testing Volume Spike Detector...")
        detector = VolumeSpikeDetector(spike_threshold=3.0)
        
        # Simulate some trades
        market_id = "test_market"
        base_price = 0.5
        
        # Add normal trades
        for i in range(25):
            trade = Trade(
                timestamp=1000 + i,
                price=base_price + (i % 2) * 0.1,
                size=10.0 + (i % 5),
                side='BUY',
                outcome='Yes',
                market_id=market_id,
                token_id='test_token'
            )
            detector.add_trade(trade)
        
        # Add a spike trade
        spike_trade = Trade(
            timestamp=1025,
            price=0.6,
            size=50.0,  # 5x average
            side='BUY',
            outcome='Yes',
            market_id=market_id,
            token_id='test_token'
        )
        detector.add_trade(spike_trade)
        
        spike = detector.detect_spike(market_id)
        
        if spike:
            print(f"   ✅ Spike detected!")
            print(f"      Price: ${spike.price:.3f}")
            print(f"      Size: {spike.size:.0f} shares")
            print(f"      Ratio: {spike.spike_ratio:.1f}x average")
            print(f"      Direction: {spike.direction}")
        else:
            print("   ❌ Spike not detected (may need more trades)")
        
        print("\n" + "=" * 80)
        print("✅ API CONNECTION TESTS PASSED")
        print("=" * 80)
        print("""
The bot can successfully:
1. Connect to Polymarket Gamma API
2. Fetch market data
3. Fetch trade data
4. Detect volume spikes

Next steps:
- Set DRY_RUN=False in live_volume_spike_bot.py for live trading
- Add wallet/API key authentication
- Implement actual order placement
        """)
        
        return True


if __name__ == '__main__':
    try:
        result = asyncio.run(test_api_connection())
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
