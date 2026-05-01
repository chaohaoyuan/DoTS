#!/usr/bin/env python
"""Run multi-objective Optuna search for optimal task vector merge coefficients.

Searches for scale_a and scale_b that maximize consistency (via vLLM) while
minimizing perplexity on a sampled dataset. Uses NSGA-II multi-objective
optimization over the Pareto front.

Usage:
    python scripts/run_search.py --config configs/search/sft_grpo.json --dataset-path data/sampled.parquet
"""
from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Pareto search for DoTS coefficient selection.")
    parser.add_argument("--config", required=True, help="Path to a search config JSON file.")
    parser.add_argument("--dataset-path", required=True, help="Path to a sampled parquet file.")
    args = parser.parse_args()
    from dots.config import load_config
    from dots.search import run_search

    run_search(load_config(args.config), dataset_path=args.dataset_path)


if __name__ == "__main__":
    main()
