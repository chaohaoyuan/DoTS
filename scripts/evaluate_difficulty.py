#!/usr/bin/env python
"""Evaluate sample-wise consistency and derive difficulty scores.

This script applies each sparse task vector to the base model in turn,
generates multiple responses per sample via vLLM, and computes a
difficulty score = 1 - avg_consistency across models.

Usage:
    python scripts/evaluate_difficulty.py --config configs/difficulty/sft_grpo.json
"""
from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sample-wise consistency and derive difficulty scores.")
    parser.add_argument("--config", required=True, help="Path to a difficulty config JSON file.")
    args = parser.parse_args()
    from dots.config import load_config
    from dots.difficulty import run_difficulty_eval

    run_difficulty_eval(load_config(args.config))


if __name__ == "__main__":
    main()
