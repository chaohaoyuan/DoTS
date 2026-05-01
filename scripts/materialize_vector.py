#!/usr/bin/env python
"""Materialize sparse task vectors into standalone model checkpoints.

Creates sparse task vectors at specified ratios (via keep_top_k_abs),
optionally rescales them, and saves the resulting merged checkpoints.

Usage:
    python scripts/materialize_vector.py --config configs/materialize/sft.json
"""
from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize sparsified task vectors into checkpoints.")
    parser.add_argument("--config", required=True, help="Path to a vector materialization config JSON file.")
    args = parser.parse_args()
    from dots.config import load_config
    from dots.materialize import run_materialization

    run_materialization(load_config(args.config))


if __name__ == "__main__":
    main()
