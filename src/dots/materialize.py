from __future__ import annotations

import gc
from pathlib import Path

from transformers import AutoTokenizer

from dots.config import dump_json, ensure_dir
from dots.task_vector import TaskVector


def run_materialization(config: dict) -> list[Path]:
    base_model_path = config["base_model_path"]
    source_model_path = config["source_model_path"]
    source_label = config.get("source_label", Path(source_model_path).name)
    fixed_scale = float(config.get("fixed_scale", 1.0))
    target_ratios = config.get("target_ratios", [0.3])
    output_dir = ensure_dir(config["output_dir"])
    cache_dir = config.get("cache_dir")
    rescale_to_original_norm = bool(config.get("rescale_to_original_norm", False))

    merged_dir = ensure_dir(output_dir / "merged_models")
    manifest_path = output_dir / "materialized_models.json"

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    full_vector = TaskVector(
        pretrained_checkpoint=base_model_path,
        finetuned_checkpoint=source_model_path,
        cache_dir=cache_dir,
    )
    original_norm = full_vector.norm()
    outputs: list[Path] = []
    manifest: list[dict] = []

    for ratio in target_ratios:
        vector = full_vector.keep_top_k_abs(float(ratio))
        scale_value = fixed_scale
        rescale_factor = 1.0
        if rescale_to_original_norm:
            vector, rescale_factor = vector.rescale_to_norm(original_norm)
            scale_value = rescale_factor

        top_percent = int(float(ratio) * 100)
        model_name = f"{source_label}_scale{scale_value}_top{top_percent}"
        if rescale_to_original_norm:
            model_name += "_rescaled"

        save_path = merged_dir / model_name
        if save_path.exists():
            print(f"Model already exists, skipping: {save_path}")
            outputs.append(save_path)
            continue

        print(f"Materializing ratio={ratio} to: {save_path}")
        merged_model = vector.apply_to(base_model_path, scaling_coef=fixed_scale)
        merged_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        manifest.append(
            {
                "ratio": ratio,
                "rescale_factor": rescale_factor,
                "fixed_scale": fixed_scale,
                "output_path": str(save_path),
            }
        )
        outputs.append(save_path)

        del merged_model
        del vector
        gc.collect()

    dump_json(manifest, manifest_path)
    print(f"Saved materialization manifest to: {manifest_path}")
    return outputs
