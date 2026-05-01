# Decouple before Integration: Test-time Synthesis of SFT and RLVR Task Vectors

Official code for the paper **"Decouple before Integration: Test-time Synthesis of SFT and RLVR Task Vectors"**.

Huggingface Checkpoint: https://huggingface.co/collections/ychaohao/dots

## Overview

DoTS merges multiple fine-tuned LLMs by combining their **sparse task vectors** — the parameter deltas between a base model and its fine-tuned variants. The core idea is:

1. **Sparsify** each task vector by keeping only the top-30% parameters with the largest absolute values (magnitude-based pruning per-layer).
2. **Merge** the sparse task vectors with learned coefficients: `merged = scale_a * sparse_vector_a + scale_b * sparse_vector_b`.
3. **Optimize** the merge coefficients via multi-objective Optuna search, balancing consistency and perplexity on a small validation set.

This approach preserves each model's specialized capabilities while minimizing interference, outperforming both individual models and dense (unpruned) task vector merging.

## Installation

```bash
git clone <repo-url>
cd dots
pip install -e .
```

Additional dependencies:
- [vLLM](https://github.com/vllm-project/vllm) for efficient LLM inference during evaluation.

## Data

The `data/` directory contains the preprocessed datasets used in the paper:

| File | Description |
|------|-------------|
| `valid.all.dedup.parquet` | Deduplicated dataset for difficulty scoring and sampling |
| `valid.all.parquet` | Full validation set for merge evaluation |
| `valid.arc_c.parquet` | ARC-Challenge subset |
| `valid.gpqa.parquet` | GPQA subset |
| `valid.mmlu_pro.parquet` | MMLU-Pro subset |
| `valid.parquet` | Original validation set |

The deduplication was performed using exact-match on prompt text, keeping only the first occurrence of each unique prompt.

## Architecture

```
data/
├── valid.all.dedup.parquet
├── valid.all.parquet
├── valid.arc_c.parquet
├── valid.gpqa.parquet
├── valid.mmlu_pro.parquet
└── valid.parquet

src/dots/
├── task_vector.py    # TaskVector: compute, sparsify (keep_top_k_abs), apply
├── search.py         # Multi-objective Optuna search for merge coefficients
├── difficulty.py     # Sample-wise difficulty scoring via consistency
├── sampling.py       # Easy/hard data subset sampling
├── merge_eval.py     # Merge vectors + run external evaluation
├── materialize.py    # Materialize sparse task vectors at various ratios
├── vllm_eval.py      # vLLM-based consistency evaluation
├── metrics.py        # Perplexity calculation
├── config.py         # Config loading with path resolution
├── text_utils.py     # Prompt parsing, answer extraction
└── results.py        # Score parsing from eval output

scripts/
├── evaluate_difficulty.py   # Step 1: Score sample difficulty
├── sample_dataset.py        # Step 2: Sample easy/hard subsets
├── run_search.py            # Step 3: Search merge coefficients
├── run_merge_eval.py        # Step 4: Merge & evaluate
└── materialize_vector.py    # Utility: Materialize sparse checkpoints
```

## Pipeline

### Step 1: Difficulty Evaluation

Apply each sparse task vector to the base model, generate multiple responses per sample, and compute a difficulty score based on answer consistency.

```bash
python scripts/evaluate_difficulty.py --config configs/difficulty/sft_grpo.json
```

Output: `output/difficulty/sft_grpo/consistency_difficulty_details.json` — per-sample difficulty scores.

### Step 2: Data Sampling

Split scored samples at the median into easy/hard pools and randomly sample from each.

```bash
python scripts/sample_dataset.py --config configs/sampling/sft_grpo.json
```

Output: `output/sampling/sft_grpo/sampled_easy_hard_dataset_*_seed_0.parquet`

### Step 3: Merge Coefficient Search

Multi-objective Optuna search (NSGA-II) to find optimal `scale_a`, `scale_b` values.

```bash
python scripts/run_search.py \
  --config configs/search/sft_grpo.json \
  --dataset-path output/sampling/sft_grpo/sampled_easy_hard_dataset_16_seed_0.parquet
```

Output: `output/search/sft_grpo/optuna_results_*.csv` with Pareto-optimal coefficients.

### Step 4: Merge & Evaluate

Merge two sparse task vectors with the chosen coefficients and run full evaluation.

```bash
python scripts/run_merge_eval.py \
  --config configs/merge_eval/sft_grpo.json \
  --scale-a 1.2 --scale-b 0.8
```

## Core API

```python
from dots.task_vector import TaskVector

# Compute task vector: delta = finetuned - base
vector = TaskVector(
    pretrained_checkpoint="path/to/base_model",
    finetuned_checkpoint="path/to/SFT_model",
)

# Sparsify: keep top 30% parameters by absolute magnitude
sparse_vector = vector.keep_top_k_abs(0.3)

# Merge two sparse vectors
merged = scale_a * sparse_vector_a + scale_b * sparse_vector_b

# Apply merged vector to base model
model = merged.apply_to("path/to/base_model")
```

## Configuration

All pipeline steps use JSON config files. Key fields:

| Field | Description |
|-------|-------------|
| `base_model_path` | Path to the base/pretrained model |
| `model_a` / `model_b` | `{"label": "...", "path": "..."}` for each fine-tuned model |
| `votes` | Number of generations per sample for consistency scoring |
| `trials` | Number of Optuna trials for coefficient search |
| `search_space` | `{"a_min", "a_max", "b_min", "b_max"}` for merge coefficient search |
| `sample_size_per_group` | Samples per easy/hard group |
| `cache_dir` | Directory for cached task vectors |

Sparsity ratio defaults to `0.3` (30%) and can be overridden via config key `top_ratio` per model or via environment variable `DOTS_SPARSITY_RATIO`.

## Model Pairs

The configs support two model pairs from the paper:
- **SFT + GRPO** (`configs/{step}/sft_grpo.json`)
- **ExGRPO + ReLIFT** (`configs/{step}/exgrpo_relift.json`)

## Citation

If you use this code in your research, please cite:

```bibtex
```

## License

MIT License.
