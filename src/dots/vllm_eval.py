from __future__ import annotations

import json
import multiprocessing as mp
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from dots.text_utils import build_reasoning_prompt, extract_answer, parse_prompt_text


def run_vllm_worker(
    model_path: str,
    prompts: list[str],
    sample_count: int,
    queue,
    result_file: str,
    gpu_count: int,
    generation_config: dict[str, Any],
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(gpu_count)))
    try:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=model_path,
            tensor_parallel_size=gpu_count,
            gpu_memory_utilization=generation_config.get("gpu_memory_utilization", 0.90),
            trust_remote_code=True,
            dtype=generation_config.get("dtype", "float16"),
        )

        sampling_params = SamplingParams(
            temperature=generation_config.get("temperature", 0.6),
            top_p=generation_config.get("top_p", 1.0),
            max_tokens=generation_config.get("max_tokens", 8192),
            n=sample_count,
        )
        outputs = llm.generate(prompts, sampling_params)
        results = [[candidate.text for candidate in output.outputs] for output in outputs]
        with open(result_file, "w", encoding="utf-8") as handle:
            json.dump(results, handle)
        queue.put({"status": "success"})
    except Exception as exc:
        queue.put({"status": "error", "message": str(exc)})


def calculate_consistency_score_vllm(
    model_path: str | Path,
    df_samples,
    sample_count: int = 3,
    generation_config: dict[str, Any] | None = None,
    timeout_seconds: int = 3600,
) -> float:
    scores = get_consistency_per_sample(
        model_path=model_path,
        df_samples=df_samples,
        sample_count=sample_count,
        generation_config=generation_config,
        timeout_seconds=timeout_seconds,
    )
    if not scores:
        return 0.0
    return sum(scores.values()) / len(scores)


def get_consistency_per_sample(
    model_path: str | Path,
    df_samples,
    sample_count: int = 5,
    generation_config: dict[str, Any] | None = None,
    timeout_seconds: int = 7200,
) -> dict[int, float]:
    generation = generation_config or {}
    prompts: list[str] = []
    valid_indices: list[int] = []

    for index, row in df_samples.iterrows():
        question = parse_prompt_text(row)
        if not question.strip():
            continue
        prompts.append(build_reasoning_prompt(question))
        valid_indices.append(index)

    if not prompts:
        return {}

    gpu_count = torch.cuda.device_count()
    if gpu_count <= 0:
        raise RuntimeError("vLLM evaluation requires at least one CUDA device.")

    context = mp.get_context("spawn")
    queue = context.Queue()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as handle:
        result_path = handle.name

    try:
        process = context.Process(
            target=run_vllm_worker,
            args=(str(model_path), prompts, sample_count, queue, result_path, gpu_count, generation),
        )
        process.start()
        status = queue.get(timeout=timeout_seconds)
        process.join()

        if status["status"] != "success":
            raise RuntimeError(status.get("message", "Unknown vLLM error"))

        with open(result_path, "r", encoding="utf-8") as handle:
            all_responses = json.load(handle)
    finally:
        if os.path.exists(result_path):
            os.remove(result_path)

    consistency_map: dict[int, float] = {}
    for index, responses in zip(valid_indices, all_responses):
        counts = Counter(extract_answer(response) for response in responses)
        if not counts:
            consistency_map[index] = 0.0
            continue
        answer, hits = counts.most_common(1)[0]
        consistency_map[index] = 0.0 if answer == "No_Answer" else hits / sample_count

    return consistency_map
