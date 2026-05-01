from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dots.config import ensure_dir


def run_sampling(config: dict) -> list[Path]:
    original_data_path = Path(config["original_data_path"])
    score_file_path = Path(config["score_file_path"])
    output_dir = ensure_dir(config["output_dir"])

    with score_file_path.open("r", encoding="utf-8") as handle:
        score_data = json.load(handle)

    score_frame = pd.DataFrame(score_data)
    original_frame = pd.read_parquet(original_data_path)

    filtered = score_frame[score_frame["difficulty_score"] <= config.get("diff_threshold", 0.8)].copy()
    ordered = filtered.sort_values(by="difficulty_score", ascending=True).reset_index(drop=True)

    midpoint = len(ordered) // 2
    easy_pool = ordered.iloc[:midpoint]
    hard_pool = ordered.iloc[midpoint:]

    sample_size = int(config.get("sample_size_per_group", 8))
    seeds = config.get("seeds", [0])
    outputs: list[Path] = []

    for seed in seeds:
        easy_count = min(sample_size, len(easy_pool))
        hard_count = min(sample_size, len(hard_pool))

        sampled_easy = easy_pool.sample(n=easy_count, random_state=seed).copy()
        sampled_easy["split_label"] = "easy"
        sampled_hard = hard_pool.sample(n=hard_count, random_state=seed).copy()
        sampled_hard["split_label"] = "hard"

        selected_indices = pd.concat([sampled_easy, sampled_hard], ignore_index=True)
        target_indices = selected_indices["original_index"].tolist()

        output_frame = original_frame.loc[target_indices].copy()
        if len(output_frame) > len(target_indices):
            output_frame = output_frame[~output_frame.index.duplicated(keep="first")]

        label_map = selected_indices.set_index("original_index")[["split_label", "difficulty_score"]]
        output_frame = output_frame.join(label_map)

        total_count = len(selected_indices)
        file_name = f"sampled_easy_hard_dataset_{total_count}_seed_{seed}.parquet"
        output_path = output_dir / file_name
        output_frame.to_parquet(output_path)
        outputs.append(output_path)
        print(f"Saved sampled dataset to: {output_path}")

    return outputs
