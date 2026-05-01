from __future__ import annotations

import gc
import os
import subprocess
from pathlib import Path

import torch
from transformers import AutoTokenizer

from dots.config import ensure_dir
from dots.results import parse_eval_score
from dots.task_vector import build_task_vector


def _log(message: str) -> None:
    print(f"[merge_eval] {message}", flush=True)


def _run_with_live_output(command: list[str], env: dict[str, str], log_file: Path) -> tuple[int, str]:
    command_text = " ".join(command)
    _log(f"Launching evaluation command: {command_text}")
    _log(f"Streaming evaluation output to terminal and log file: {log_file}")

    output_chunks: list[str] = []
    with log_file.open("w", encoding="utf-8") as handle:
        handle.write(f"COMMAND:\n{command_text}\n\nOUTPUT:\n")
        handle.flush()

        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(f"[eval] {line}", end="", flush=True)
            handle.write(line)
            handle.flush()
            output_chunks.append(line)

        return_code = process.wait()
        handle.write(f"\n[exit_code] {return_code}\n")
        handle.flush()

    return return_code, "".join(output_chunks)


def run_merge_eval(config: dict, scale_a: float, scale_b: float, dataset_path: str | None = None) -> tuple[Path, float | None]:
    base_model_path = config["base_model_path"]
    data_file = dataset_path or config["data_file"]
    output_dir = ensure_dir(config["output_dir"])
    merged_dir = ensure_dir(output_dir / "merged_models")
    log_dir = ensure_dir(output_dir / "eval_logs")
    cache_dir = config.get("cache_dir")

    model_name_template = config.get("model_name_template", "merge_a_{scale_a}_b_{scale_b}")
    model_name = model_name_template.format(scale_a=scale_a, scale_b=scale_b)
    save_path = merged_dir / model_name
    log_file = log_dir / f"{model_name}.log"
    jsonl_output = log_dir / f"{model_name}.jsonl"

    _log(f"Starting merge evaluation for model: {model_name}")
    _log(f"scale_a={scale_a}, scale_b={scale_b}")
    _log(f"base_model={base_model_path}")
    _log(f"evaluation_data={data_file}")
    _log(f"merged_model_dir={save_path}")

    _log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    _log(f"Building task vector for model_a: {config['model_a'].get('label', 'model_a')}")
    vector_a, _ = build_task_vector(base_model_path, config["model_a"], cache_dir=cache_dir)
    _log(f"Building task vector for model_b: {config['model_b'].get('label', 'model_b')}")
    vector_b, _ = build_task_vector(base_model_path, config["model_b"], cache_dir=cache_dir)

    if not save_path.exists():
        _log("Merged checkpoint not found. Materializing merged model on CPU...")
        combined = (vector_a * scale_a) + (vector_b * scale_b)
        merged_model = combined.apply_to(base_model_path, device_map=None)
        merged_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        del merged_model
        del combined
        _log(f"Merged checkpoint saved to: {save_path}")
    else:
        _log(f"Reusing existing merged checkpoint: {save_path}")

    del vector_a
    del vector_b
    del tokenizer
    _log("Cleaning up in-memory objects before evaluation...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    command = [
        "python",
        config["eval_script"],
        "--model_path",
        str(save_path),
        "--input_file",
        str(data_file),
        "--remove_system",
        str(config.get("remove_system", True)),
        "--add_oat_evaluate",
        str(config.get("add_oat_evaluate", True)),
        "--output_file",
        str(jsonl_output),
        "--template",
        config.get("template", "own"),
    ]

    env = os.environ.copy()
    if config.get("cuda_visible_devices"):
        env["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]
    env["VLLM_DISABLE_CUSTOM_ALL_REDUCE"] = "1"

    _log("Starting benchmark / evaluation...")
    return_code, combined_output = _run_with_live_output(command, env=env, log_file=log_file)

    if return_code != 0:
        raise RuntimeError(f"Evaluation failed for {model_name}. See log: {log_file}")

    score = parse_eval_score(combined_output)
    _log(f"Merge evaluation finished for {model_name}. Score={score}")
    _log(f"Logs written to: {log_file}")
    return save_path, score
