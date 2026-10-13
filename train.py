"""Fine-tune a DistilBERT classifier on the BANKING77 dataset."""

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import ClassLabel, load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        default="distilbert-base-uncased",
        help="Hugging Face model checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/banking77-distilbert",
        help="Directory for the trained model, tokenizer, and metrics.",
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def compute_metrics(prediction):
    logits, labels = prediction
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
    }


def main():
    args = parse_args()
    if args.max_length < 1:
        raise ValueError("--max-length must be a positive integer.")
    if not 0 < args.validation_size < 1:
        raise ValueError("--validation-size must be between 0 and 1.")

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(__file__).resolve().parent / "data"
    train_csv = data_dir / "banking77_train.csv"
    test_csv = data_dir / "banking77_test.csv"
    if not train_csv.is_file() or not test_csv.is_file():
        raise FileNotFoundError(
            "BANKING77 CSV files are missing. Generate them first with "
            "`.venv\\Scripts\\python.exe export_dataset.py`."
        )

    dataset = load_dataset(
        "csv",
        data_files={"train": str(train_csv), "test": str(test_csv)},
    )
    label_names = sorted(set(dataset["train"]["intent"]))
    if len(label_names) != 77:
        raise ValueError(
            f"Expected 77 BANKING77 intents in the training CSV, got {len(label_names)}."
        )
    label2id = {name: index for index, name in enumerate(label_names)}
    dataset = dataset.map(
        lambda batch: {"label": [label2id[intent] for intent in batch["intent"]]},
        batched=True,
    )
    dataset = dataset.remove_columns("intent")
    dataset = dataset.cast_column("label", ClassLabel(names=label_names))
    train_validation = dataset["train"].train_test_split(
        test_size=args.validation_size,
        seed=args.seed,
        stratify_by_column="label",
    )
    id2label = {index: name for index, name in enumerate(label_names)}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    tokenized = train_validation.map(tokenize, batched=True)
    tokenized_test = dataset["test"].map(tokenize, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    test_metrics = trainer.evaluate(tokenized_test, metric_key_prefix="test")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    trainer.save_metrics("test", test_metrics)
    trainer.save_state()

    with (output_dir / "label_names.json").open("w", encoding="utf-8") as file:
        json.dump(label_names, file, indent=2)
    print(f"Test metrics: {json.dumps(test_metrics, indent=2)}")
    print(f"Model and tokenizer saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
