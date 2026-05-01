from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM


def load_checkpoint(path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint_dir = Path(path)
    safetensors_index = checkpoint_dir / "model.safetensors.index.json"
    safetensors_single = checkpoint_dir / "model.safetensors"
    pytorch_index = checkpoint_dir / "pytorch_model.bin.index.json"
    pytorch_single = checkpoint_dir / "pytorch_model.bin"

    if safetensors_index.exists():
        state_dict: dict[str, torch.Tensor] = {}
        with safetensors_index.open("r", encoding="utf-8") as handle:
            index_data = json.load(handle)
        shard_files = sorted(set(index_data["weight_map"].values()))
        for shard_file in tqdm(shard_files, desc=f"Loading shards from {checkpoint_dir.name}"):
            state_dict.update(load_file(str(checkpoint_dir / shard_file), device="cpu"))
        return state_dict

    if safetensors_single.exists():
        return load_file(str(safetensors_single), device="cpu")

    if pytorch_index.exists():
        state_dict = {}
        with pytorch_index.open("r", encoding="utf-8") as handle:
            index_data = json.load(handle)
        shard_files = sorted(set(index_data["weight_map"].values()))
        for shard_file in tqdm(shard_files, desc=f"Loading shards from {checkpoint_dir.name}"):
            state_dict.update(torch.load(checkpoint_dir / shard_file, map_location="cpu"))
        return state_dict

    if pytorch_single.exists():
        return torch.load(pytorch_single, map_location="cpu")

    raise FileNotFoundError(f"No model weights found in {checkpoint_dir}")


class TaskVector:
    def __init__(
        self,
        pretrained_checkpoint: str | Path | None = None,
        finetuned_checkpoint: str | Path | None = None,
        vector: dict[str, torch.Tensor] | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        if vector is not None:
            self.vector = vector
            return

        if pretrained_checkpoint is None or finetuned_checkpoint is None:
            raise ValueError("Either `vector` or both checkpoint paths must be provided.")

        cache_path = None
        if cache_dir is not None:
            cache_path = self._build_cache_path(pretrained_checkpoint, finetuned_checkpoint, cache_dir)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if cache_path.exists():
                print(f"Loading cached task vector from: {cache_path}")
                self.vector = load_file(str(cache_path), device="cpu")
                return

        print(f"Loading base model: {pretrained_checkpoint}")
        pretrained_state_dict = load_checkpoint(pretrained_checkpoint)
        print(f"Loading finetuned model: {finetuned_checkpoint}")
        finetuned_state_dict = load_checkpoint(finetuned_checkpoint)

        self.vector = {}
        common_keys = set(pretrained_state_dict.keys()) & set(finetuned_state_dict.keys())
        for key in tqdm(common_keys, desc="Calculating task vector"):
            base_tensor = pretrained_state_dict[key]
            tuned_tensor = finetuned_state_dict[key]
            if base_tensor.dtype in {torch.int64, torch.uint8, torch.int32, torch.bool}:
                continue
            if base_tensor.shape != tuned_tensor.shape:
                continue
            self.vector[key] = tuned_tensor.to(torch.float32) - base_tensor.to(torch.float32)

        del pretrained_state_dict
        del finetuned_state_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if cache_path is not None:
            print(f"Saving task vector cache to: {cache_path}")
            save_file(self.vector, str(cache_path))

    @staticmethod
    def _build_cache_path(
        pretrained_checkpoint: str | Path,
        finetuned_checkpoint: str | Path,
        cache_dir: str | Path,
    ) -> Path:
        base_name = Path(finetuned_checkpoint).name
        digest = hashlib.sha1(f"{pretrained_checkpoint}::{finetuned_checkpoint}".encode("utf-8")).hexdigest()[:12]
        return Path(cache_dir) / f"{base_name}.{digest}.safetensors"

    def __add__(self, other: "TaskVector") -> "TaskVector":
        merged: dict[str, torch.Tensor] = {}
        all_keys = set(self.vector.keys()) | set(other.vector.keys())
        for key in all_keys:
            merged[key] = self.vector.get(key, torch.tensor(0.0)) + other.vector.get(key, torch.tensor(0.0))
        return TaskVector(vector=merged)

    def __radd__(self, other: Any) -> "TaskVector":
        if other in {None, 0}:
            return self
        return self.__add__(other)

    def __neg__(self) -> "TaskVector":
        return TaskVector(vector={key: -value for key, value in self.vector.items()})

    def __mul__(self, scalar: float) -> "TaskVector":
        if not isinstance(scalar, (int, float)):
            raise ValueError("TaskVector can only be multiplied by a scalar.")
        return TaskVector(vector={key: value * scalar for key, value in self.vector.items()})

    def __rmul__(self, scalar: float) -> "TaskVector":
        return self.__mul__(scalar)

    def norm(self) -> torch.Tensor:
        dot_product = torch.tensor(0.0)
        for value in self.vector.values():
            dot_product += torch.sum(value ** 2)
        return torch.sqrt(dot_product)

    def keep_top_k_abs(self, ratio: float) -> "TaskVector":
        """Keep the top `ratio` fraction of parameters with the largest absolute values.

        This is the core sparsification operation: for each layer, only the top-k
        parameters by absolute magnitude are retained; the rest are zeroed out.
        """
        if ratio >= 1.0:
            return TaskVector(vector={key: value.clone() for key, value in self.vector.items()})

        print(f"Keeping top {ratio * 100:.0f}% parameters per layer.")
        new_vector: dict[str, torch.Tensor] = {}
        for key, tensor in tqdm(self.vector.items(), desc="Pruning task vector"):
            if tensor.numel() == 0:
                new_vector[key] = tensor
                continue

            k_value = int(tensor.numel() * ratio)
            if k_value < 1:
                new_vector[key] = torch.zeros_like(tensor)
                continue

            flat_abs = tensor.abs().flatten()
            top_values, _ = torch.topk(flat_abs, k_value)
            threshold = top_values[-1]
            new_vector[key] = tensor * (tensor.abs() >= threshold)

        return TaskVector(vector=new_vector)

    def rescale_to_norm(self, target_norm: torch.Tensor | float) -> tuple["TaskVector", float]:
        target = target_norm.item() if isinstance(target_norm, torch.Tensor) else float(target_norm)
        current = self.norm().item()
        if current == 0:
            return self, 1.0
        factor = target / current
        return self * factor, factor

    def apply_to(
        self,
        pretrained_model_path: str | Path,
        scaling_coef: float = 1.0,
        device_map: str | dict[str, Any] | None = "auto",
        torch_dtype: str | torch.dtype = "auto",
    ) -> AutoModelForCausalLM:
        model = AutoModelForCausalLM.from_pretrained(
            str(pretrained_model_path),
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        state_dict = model.state_dict()
        new_state_dict = {}
        for key, base_value in state_dict.items():
            delta = self.vector.get(key)
            if delta is None:
                new_state_dict[key] = base_value
                continue
            new_state_dict[key] = base_value + scaling_coef * delta.to(base_value.device).to(base_value.dtype)
        model.load_state_dict(new_state_dict)
        return model


def apply_vector_inplace(model: AutoModelForCausalLM, vector: TaskVector) -> None:
    with torch.no_grad():
        for key, value in vector.vector.items():
            if key in model.state_dict():
                model.state_dict()[key].add_(value.to(model.state_dict()[key].device))


def remove_vector_inplace(model: AutoModelForCausalLM, vector: TaskVector) -> None:
    with torch.no_grad():
        for key, value in vector.vector.items():
            if key in model.state_dict():
                model.state_dict()[key].sub_(value.to(model.state_dict()[key].device))


def build_task_vector(
    base_model_path: str | Path,
    model_config: dict[str, Any],
    cache_dir: str | Path | None = None,
) -> tuple[TaskVector, dict[str, Any]]:
    vector = TaskVector(
        pretrained_checkpoint=base_model_path,
        finetuned_checkpoint=model_config["path"],
        cache_dir=cache_dir,
    )
    metadata: dict[str, Any] = {
        "path": model_config["path"],
        "label": model_config.get("label", "model"),
    }

    top_ratio = model_config.get("top_ratio")
    if top_ratio is None:
        top_ratio = int(os.environ.get("DOTS_SPARSITY_RATIO", "0.3"))
    top_ratio = float(top_ratio)

    if top_ratio < 1.0:
        original_norm = vector.norm()
        vector = vector.keep_top_k_abs(top_ratio)
        metadata["top_ratio"] = top_ratio
        if model_config.get("rescale_to_original_norm"):
            vector, factor = vector.rescale_to_norm(original_norm)
            metadata["rescale_factor"] = factor

    return vector, metadata
