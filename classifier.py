"""Shared model-loading and intent-prediction helpers."""

from pathlib import Path
from typing import Sequence

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def load_classifier(
    model_dir: str | Path,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel, torch.device]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_intents(
    texts: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    device: torch.device,
    top_k: int = 3,
    batch_size: int = 32,
) -> list[list[dict[str, str | float]]]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if not texts or any(not text.strip() for text in texts):
        raise ValueError("Provide a non-empty customer message.")

    results = []
    for start in range(0, len(texts), batch_size):
        inputs = tokenizer(
            list(texts[start : start + batch_size]),
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            probabilities = torch.softmax(model(**inputs).logits, dim=-1).cpu()

        for scores in probabilities:
            top_scores, top_indices = torch.topk(scores, k=min(top_k, scores.numel()))
            results.append(
                [
                    {
                        "intent": model.config.id2label[index.item()].replace("_", " "),
                        "confidence": round(score.item(), 4),
                    }
                    for score, index in zip(top_scores, top_indices)
                ]
            )
    return results
