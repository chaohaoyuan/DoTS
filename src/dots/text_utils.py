from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import numpy as np


def parse_prompt_text(row: Any) -> str:
    raw_data = row.get("prompt", None)
    problem_text = ""

    if isinstance(raw_data, np.ndarray):
        raw_data = raw_data.tolist()

    if isinstance(raw_data, str):
        raw_data = raw_data.strip()
        if raw_data.startswith("[") and raw_data.endswith("]"):
            try:
                raw_data = ast.literal_eval(raw_data)
            except (SyntaxError, ValueError):
                pass

    if isinstance(raw_data, list):
        for message in raw_data:
            if isinstance(message, dict) and message.get("role") == "user":
                problem_text = message.get("content", "")
                break

        if not problem_text and raw_data and isinstance(raw_data[-1], dict):
            problem_text = raw_data[-1].get("content", "")
    elif isinstance(raw_data, str):
        problem_text = raw_data

    return problem_text


def build_reasoning_prompt(question: str) -> str:
    return (
        "<|im_start|>user\n"
        f"{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n"
        "<|im_end|>\n<|im_start|>assistant\n"
    )


def build_ppl_prompt(question: str) -> str:
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


def extract_answer(text: str) -> str:
    stripped = text.strip()
    if "<answer>" in stripped and "</answer>" in stripped:
        try:
            return stripped.split("<answer>")[1].split("</answer>")[0].strip()
        except IndexError:
            pass

    boxed = re.findall(r"\\boxed\{([^}]+)\}", stripped)
    if boxed:
        return boxed[-1].strip()

    match = re.search(r"(?:The answer is|Answer:|Result:)\s*(-?\d+(?:\.\d+)?)", stripped, re.IGNORECASE)
    if match:
        return match.group(1)

    all_numbers = re.findall(r"-?\d+(?:\.\d+)?", stripped)
    if all_numbers:
        return all_numbers[-1]

    return "No_Answer"


def dataset_metadata_from_path(dataset_path: str | Path) -> tuple[str, str]:
    filename = Path(dataset_path).name
    match = re.search(r"_(\d+)_seed_(\d+)\.parquet$", filename)
    if match:
        return match.group(1), match.group(2)
    return "unknown", "unknown"
