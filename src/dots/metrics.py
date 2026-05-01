from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from dots.text_utils import build_ppl_prompt, parse_prompt_text


def calculate_perplexity(
    model,
    tokenizer,
    df_samples,
    device: torch.device,
    max_length: int = 8192,
    batch_size: int = 2,
) -> float:
    model.eval()
    nlls: list[float] = []
    texts: list[str] = []

    for _, row in df_samples.iterrows():
        prompt_text = parse_prompt_text(row)
        if prompt_text.strip():
            texts.append(build_ppl_prompt(prompt_text))

    if not texts:
        return 9999.0

    loss_fn = CrossEntropyLoss(reduction="none")
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs.input_ids[..., 1:].contiguous()
            shift_attention_mask = inputs.attention_mask[..., 1:].contiguous()

            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_labels.size())
            valid_loss = torch.sum(loss * shift_attention_mask, dim=1)
            valid_len = torch.sum(shift_attention_mask, dim=1)
            batch_nlls = valid_loss / valid_len
            nlls.extend(batch_nlls.tolist())

    if not nlls:
        return 9999.0

    return math.exp(float(np.mean(nlls)))
