from __future__ import annotations

import gc
import shutil
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dots.config import dump_json, ensure_dir
from dots.task_vector import apply_vector_inplace, build_task_vector, remove_vector_inplace
from dots.text_utils import parse_prompt_text
from dots.vllm_eval import get_consistency_per_sample


def run_difficulty_eval(config: dict) -> tuple[Path, Path]:
    data_file = Path(config["data_file"])
    output_dir = ensure_dir(config["output_dir"])
    temp_model_dir = Path(config["temp_model_dir"])
    cache_dir = config.get("cache_dir")
    sample_count = int(config.get("votes", 5))
    generation_config = config.get("generation", {})

    data_frame = pd.read_parquet(data_file)
    tokenizer = AutoTokenizer.from_pretrained(config["base_model_path"], trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        config["base_model_path"],
        device_map=None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_model.to(device)

    vector_a, meta_a = build_task_vector(config["base_model_path"], config["model_a"], cache_dir=cache_dir)
    vector_b, meta_b = build_task_vector(config["base_model_path"], config["model_b"], cache_dir=cache_dir)

    def evaluate_phase(task_vector, label: str) -> dict[int, float]:
        apply_vector_inplace(base_model, task_vector)
        if temp_model_dir.exists():
            shutil.rmtree(temp_model_dir)
        base_model.save_pretrained(temp_model_dir)
        tokenizer.save_pretrained(temp_model_dir)

        base_model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        scores = get_consistency_per_sample(
            model_path=temp_model_dir,
            df_samples=data_frame,
            sample_count=sample_count,
            generation_config=generation_config,
        )
        if temp_model_dir.exists():
            shutil.rmtree(temp_model_dir)

        base_model.to(device)
        remove_vector_inplace(base_model, task_vector)
        print(f"{label} average consistency: {sum(scores.values()) / max(len(scores), 1):.4f}")
        return scores

    scores_a = evaluate_phase(vector_a, meta_a["label"])
    scores_b = evaluate_phase(vector_b, meta_b["label"])

    details: list[dict] = []
    total_a = 0.0
    total_b = 0.0
    valid_count = 0

    for index, row in data_frame.iterrows():
        score_a = scores_a.get(index, 0.0)
        score_b = scores_b.get(index, 0.0)
        avg_consistency = (score_a + score_b) / 2.0
        difficulty_score = 1.0 - avg_consistency
        if index in scores_a or index in scores_b:
            total_a += score_a
            total_b += score_b
            valid_count += 1
        details.append(
            {
                "original_index": index,
                "prompt": parse_prompt_text(row),
                "consistency_a": score_a,
                "consistency_b": score_b,
                "label_a": meta_a["label"],
                "label_b": meta_b["label"],
                "avg_consistency": avg_consistency,
                "difficulty_score": difficulty_score,
            }
        )

    avg_a = total_a / valid_count if valid_count else 0.0
    avg_b = total_b / valid_count if valid_count else 0.0
    detail_path = output_dir / "consistency_difficulty_details.json"
    summary_path = output_dir / "consistency_summary.json"

    dump_json(details, detail_path)
    dump_json(
        {
            "total_samples": len(data_frame),
            "valid_samples": valid_count,
            "metrics": {
                "label_a": meta_a["label"],
                "label_b": meta_b["label"],
                "consistency_a_avg": avg_a,
                "consistency_b_avg": avg_b,
                "total_consistency_avg": (avg_a + avg_b) / 2.0,
            },
        },
        summary_path,
    )
    print(f"Saved difficulty details to: {detail_path}")
    print(f"Saved summary to: {summary_path}")
    return detail_path, summary_path
