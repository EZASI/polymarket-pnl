#!/usr/bin/env python3
"""
Continue fetching trades from offset 48000 and merge with existing data.
"""

import requests
import json
import time
from datetime import datetime

WALLET_ADDRESS = "0x1831d1abf62dfadabcabb3d1c2bf4476146a6c99"
BASE_URL = "https://data-api.polymarket.com"

def fetch_trades_with_retry(wallet_address, start_offset=48000, batch_size=500, max_retries=3):
    """Fetch trades starting from offset with retry logic."""
    all_trades = []
    offset = start_offset

    print(f"Continuing from offset {start_offset}...")

    while True:
        url = f"{BASE_URL}/trades?user={wallet_address}&limit={batch_size}&offset={offset}"
        print(f"Fetching offset {offset}...")

        for retry in range(max_retries):
            try:
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                trades = response.json()

                if not trades:
                    print(f"No more trades at offset {offset}")
                    return all_trades

                all_trades.extend(trades)
                print(f"  Got {len(trades)} trades (new total: {len(all_trades)})")

                if len(trades) < batch_size:
                    return all_trades

                offset += batch_size
                time.sleep(1)  # Rate limiting
                break

            except Exception as e:
                print(f"  Retry {retry + 1}/{max_retries}: {e}")
                time.sleep(2 ** (retry + 1))
                if retry == max_retries - 1:
                    print("Max retries reached, returning collected trades")
                    return all_trades

    return all_trades


def main():
    # Load existing data
    print("Loading existing trades...")
    with open("mty44_trades.json", 'r') as f:
        data = json.load(f)

    existing_trades = data.get('trades', [])
    print(f"Existing trades: {len(existing_trades)}")

    # Fetch remaining trades
    new_trades = fetch_trades_with_retry(WALLET_ADDRESS, start_offset=48000)
    print(f"New trades fetched: {len(new_trades)}")

    # Merge trades
    all_trades = existing_trades + new_trades
    print(f"Total trades after merge: {len(all_trades)}")

    # Update data
    data['trades'] = all_trades
    data['total_trades'] = len(all_trades)
    data['fetched_at'] = datetime.utcnow().isoformat()

    # Save updated data
    with open("mty44_trades.json", 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nUpdated data saved to mty44_trades.json")


if __name__ == "__main__":
    main()
