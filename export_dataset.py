"""Download BANKING77 and export its official splits as CSV files."""

import argparse
from pathlib import Path

import pandas as pd


EXPECTED_ROWS = {"train": 10_003, "test": 3_080}
DATASET_URLS = {
    "train": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
    "test": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory for the generated CSV files (default: data).",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, url in DATASET_URLS.items():
        frame = pd.read_csv(url)
        expected_rows = EXPECTED_ROWS[split_name]
        if len(frame) != expected_rows:
            raise ValueError(
                f"Expected {expected_rows} {split_name} examples, got {len(frame)}."
            )
        if list(frame.columns) != ["text", "category"]:
            raise ValueError(
                f"Unexpected {split_name} CSV columns: {list(frame.columns)}"
            )
        frame = frame.rename(columns={"category": "intent"})

        output_path = output_dir / f"banking77_{split_name}.csv"
        frame.to_csv(output_path, index=False, encoding="utf-8")
        print(
            f"Saved {len(frame):,} rows and "
            f"{frame['intent'].nunique()} intents to {output_path.resolve()}"
        )


if __name__ == "__main__":
    main()
