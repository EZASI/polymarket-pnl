#!/usr/bin/env python3
"""
Fetch all trades for a Polymarket user using the data API.
"""

import requests
import json
import time
from datetime import datetime

WALLET_ADDRESS = "0x1831d1abf62dfadabcabb3d1c2bf4476146a6c99"
BASE_URL = "https://data-api.polymarket.com"

def fetch_all_trades(wallet_address, batch_size=500):
    """Fetch all trades for a given wallet address."""
    all_trades = []
    offset = 0

    print(f"Fetching trades for wallet: {wallet_address}")

    while True:
        url = f"{BASE_URL}/trades?user={wallet_address}&limit={batch_size}&offset={offset}"
        print(f"Fetching offset {offset}...")

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            trades = response.json()

            if not trades:
                print(f"No more trades at offset {offset}")
                break

            all_trades.extend(trades)
            print(f"  Got {len(trades)} trades (total: {len(all_trades)})")

            if len(trades) < batch_size:
                break

            offset += batch_size
            time.sleep(0.5)  # Rate limiting

        except requests.exceptions.RequestException as e:
            print(f"Error fetching trades: {e}")
            break

    return all_trades


def fetch_positions(wallet_address):
    """Fetch current positions for a wallet."""
    url = f"{BASE_URL}/positions?user={wallet_address}"
    print(f"Fetching positions...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching positions: {e}")
        return []


def fetch_activity(wallet_address):
    """Fetch user activity."""
    url = f"{BASE_URL}/activity?user={wallet_address}"
    print(f"Fetching activity...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching activity: {e}")
        return []


def main():
    print("=" * 60)
    print("POLYMARKET TRADE FETCHER")
    print(f"Wallet: {WALLET_ADDRESS}")
    print("=" * 60)

    # Fetch all trades
    trades = fetch_all_trades(WALLET_ADDRESS)
    print(f"\nTotal trades fetched: {len(trades)}")

    # Fetch positions
    positions = fetch_positions(WALLET_ADDRESS)
    print(f"Total positions fetched: {len(positions)}")

    # Fetch activity
    activity = fetch_activity(WALLET_ADDRESS)
    print(f"Total activity records: {len(activity) if isinstance(activity, list) else 'N/A'}")

    # Save all data
    data = {
        "wallet_address": WALLET_ADDRESS,
        "fetched_at": datetime.utcnow().isoformat(),
        "total_trades": len(trades),
        "total_positions": len(positions),
        "trades": trades,
        "positions": positions,
        "activity": activity
    }

    output_file = "mty44_trades.json"
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nData saved to {output_file}")

    # Print summary
    if trades:
        print("\n" + "=" * 60)
        print("TRADE SUMMARY")
        print("=" * 60)

        # Calculate some stats
        total_volume = sum(float(t.get('size', 0)) * float(t.get('price', 0)) for t in trades)
        buy_trades = [t for t in trades if t.get('side') == 'BUY']
        sell_trades = [t for t in trades if t.get('side') == 'SELL']

        print(f"Total trades: {len(trades)}")
        print(f"Buy trades: {len(buy_trades)}")
        print(f"Sell trades: {len(sell_trades)}")
        print(f"Estimated volume: ${total_volume:,.2f}")

        # Get unique markets
        markets = set(t.get('title', 'Unknown') for t in trades)
        print(f"Unique markets traded: {len(markets)}")

        # Print first 10 and last 10 trades
        print("\n--- FIRST 10 TRADES ---")
        for t in trades[:10]:
            print(f"  {t.get('timestamp', 'N/A')} | {t.get('side', 'N/A'):4} | "
                  f"${float(t.get('price', 0)):.4f} x {float(t.get('size', 0)):,.2f} | "
                  f"{t.get('outcome', 'N/A')} | {t.get('title', 'N/A')[:50]}")

        print("\n--- LAST 10 TRADES ---")
        for t in trades[-10:]:
            print(f"  {t.get('timestamp', 'N/A')} | {t.get('side', 'N/A'):4} | "
                  f"${float(t.get('price', 0)):.4f} x {float(t.get('size', 0)):,.2f} | "
                  f"{t.get('outcome', 'N/A')} | {t.get('title', 'N/A')[:50]}")


if __name__ == "__main__":
    main()
