#!/usr/bin/env python3
"""
Extract Hugging Face dataset to local files.

Dataset: WinkingFace/CryptoLM-Bitcoin-BTC-USDT
Outputs: parquet/csv in ./data/
"""

import argparse
import os
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract HF dataset to disk.")
    parser.add_argument(
        "--dataset",
        default="WinkingFace/CryptoLM-Bitcoin-BTC-USDT",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to export (default: train)",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Output format",
    )
    parser.add_argument(
        "--out-dir",
        default="data",
        help="Output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset, split=args.split)

    out_path = out_dir / f"{args.dataset.replace('/', '_')}_{args.split}.{args.format}"
    if args.format == "parquet":
        dataset.to_parquet(str(out_path))
    else:
        dataset.to_csv(str(out_path))

    print(f"Saved {len(dataset):,} rows to {out_path}")


if __name__ == "__main__":
    main()
