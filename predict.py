"""Predict BANKING77 intents with a fine-tuned classifier."""

import argparse
import json

from classifier import load_classifier, predict_intents


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default="outputs/banking77-distilbert",
        help="Directory containing the fine-tuned model and tokenizer.",
    )
    parser.add_argument(
        "--text",
        action="append",
        required=True,
        help="Customer message to classify. Repeat to classify multiple messages.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")

    tokenizer, model, device = load_classifier(args.model_dir)
    predictions_by_text = predict_intents(
        args.text,
        tokenizer,
        model,
        device,
        top_k=args.top_k,
    )
    for text, predictions in zip(args.text, predictions_by_text):
        print(json.dumps({"text": text, "predictions": predictions}, ensure_ascii=False))


if __name__ == "__main__":
    main()
