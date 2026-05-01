#!/usr/bin/env python
"""Sample easy and hard subsets from difficulty scores.

Splits the difficulty-scored samples at the median into easy and hard pools,
then randomly samples from each pool at multiple seeds.

Usage:
    python scripts/sample_dataset.py --config configs/sampling/sft_grpo.json
"""
from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample easy and hard subsets from difficulty scores.")
    parser.add_argument("--config", required=True, help="Path to a sampling config JSON file.")
    args = parser.parse_args()
    from dots.config import load_config
    from dots.sampling import run_sampling

    run_sampling(load_config(args.config))


if __name__ == "__main__":
    main()
