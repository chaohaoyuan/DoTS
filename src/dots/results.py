from __future__ import annotations

import re

import pandas as pd


def parse_eval_score(stdout: str) -> float | None:
    patterns = [
        r"Final Accuracy: (\d+\.\d+)",
        r"\"accuracy\": (\d+\.\d+)",
        r"accuracy.*[ :]+(\d+\.\d+)",
        r"Score: (\d+\.\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, stdout, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def normalize_search_frame(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.copy()
    if "scale_a" not in renamed.columns and "sft" in renamed.columns:
        renamed = renamed.rename(columns={"sft": "scale_a"})
    if "scale_b" not in renamed.columns and "grpo" in renamed.columns:
        renamed = renamed.rename(columns={"grpo": "scale_b"})
    if "scale_a" not in renamed.columns and "left_scale" in renamed.columns:
        renamed = renamed.rename(columns={"left_scale": "scale_a"})
    if "scale_b" not in renamed.columns and "right_scale" in renamed.columns:
        renamed = renamed.rename(columns={"right_scale": "scale_b"})
    return renamed
