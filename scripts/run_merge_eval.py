#!/usr/bin/env python
"""Merge two sparse task vectors with given coefficients and evaluate.

Applies scale_a * vector_a + scale_b * vector_b to the base model, saves
the merged checkpoint, and runs an external vLLM evaluation script.

Usage:
    python scripts/run_merge_eval.py --config configs/merge_eval/sft_grpo.json --scale-a 1.2 --scale-b 0.8
"""
from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge two task vectors and evaluate the resulting checkpoint.")
    parser.add_argument("--config", required=True, help="Path to a merge-eval config JSON file.")
    parser.add_argument("--scale-a", required=True, type=float, help="Scale for model_a.")
    parser.add_argument("--scale-b", required=True, type=float, help="Scale for model_b.")
    parser.add_argument("--dataset-path", default=None, help="Optional override for the evaluation parquet file.")
    args = parser.parse_args()
    from dots.config import load_config
    from dots.merge_eval import run_merge_eval

    run_merge_eval(
        config=load_config(args.config),
        scale_a=args.scale_a,
        scale_b=args.scale_b,
        dataset_path=args.dataset_path,
    )


if __name__ == "__main__":
    main()
