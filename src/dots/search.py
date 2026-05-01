from __future__ import annotations

import gc
import json
import shutil
from pathlib import Path

import optuna
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dots.config import ensure_dir
from dots.metrics import calculate_perplexity
from dots.task_vector import apply_vector_inplace, build_task_vector, remove_vector_inplace
from dots.text_utils import dataset_metadata_from_path
from dots.vllm_eval import calculate_consistency_score_vllm


def run_search(config: dict, dataset_path: str) -> Path:
    data_path = Path(dataset_path)
    output_dir = ensure_dir(config["output_dir"])
    temp_model_dir = Path(config["temp_model_dir"])
    cache_dir = config.get("cache_dir")

    data_count, seed = dataset_metadata_from_path(data_path)
    data_frame = pd.read_parquet(data_path)
    tokenizer = AutoTokenizer.from_pretrained(config["base_model_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    votes = int(config.get("votes", 7))
    trial_count = int(config.get("trials", 100))
    search_space = config.get("search_space", {"a_min": 0.5, "a_max": 2.0, "b_min": 0.5, "b_max": 2.0})
    generation_config = config.get("generation", {})
    ppl_max_length = int(config.get("ppl_max_length", 8192))

    log_path = output_dir / f"log_optuna_{data_count}_seed_{seed}.jsonl"
    csv_path = output_dir / f"optuna_results_{data_count}_seed_{seed}.csv"

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        scale_a = trial.suggest_float("scale_a", float(search_space["a_min"]), float(search_space["a_max"]))
        scale_b = trial.suggest_float("scale_b", float(search_space["b_min"]), float(search_space["b_max"]))
        combined = (vector_a * scale_a) + (vector_b * scale_b)
        apply_vector_inplace(base_model, combined)

        consistency = 0.0
        ppl = 9999.0
        try:
            ppl = calculate_perplexity(
                model=base_model,
                tokenizer=tokenizer,
                df_samples=data_frame,
                device=device,
                max_length=ppl_max_length,
            )
            if temp_model_dir.exists():
                shutil.rmtree(temp_model_dir)
            base_model.save_pretrained(temp_model_dir)
            tokenizer.save_pretrained(temp_model_dir)

            base_model.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            consistency = calculate_consistency_score_vllm(
                model_path=temp_model_dir,
                df_samples=data_frame,
                sample_count=votes,
                generation_config=generation_config,
            )
        finally:
            if temp_model_dir.exists():
                shutil.rmtree(temp_model_dir)
            base_model.to(device)
            remove_vector_inplace(base_model, combined)

        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "trial": trial.number,
                        "data_count": data_count,
                        "seed": seed,
                        "scale_a": scale_a,
                        "scale_b": scale_b,
                        "label_a": meta_a["label"],
                        "label_b": meta_b["label"],
                        "consistency": consistency,
                        "ppl": ppl,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        print(
            f"Trial {trial.number}: "
            f"{meta_a['label']}={scale_a:.4f}, {meta_b['label']}={scale_b:.4f}, "
            f"consistency={consistency:.4f}, ppl={ppl:.2f}"
        )
        return consistency, ppl

    study = optuna.create_study(directions=["maximize", "minimize"])
    study.optimize(objective, n_trials=trial_count)

    rows = []
    for trial in study.trials:
        if not trial.values:
            continue
        rows.append(
            {
                "data_count": data_count,
                "seed": seed,
                "scale_a": trial.params["scale_a"],
                "scale_b": trial.params["scale_b"],
                "label_a": meta_a["label"],
                "label_b": meta_b["label"],
                "consistency": trial.values[0],
                "ppl": trial.values[1],
                "is_pareto": trial in study.best_trials,
            }
        )

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved Optuna results to: {csv_path}")
    return csv_path
