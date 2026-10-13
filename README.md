# BANKING77 Customer Intent Classification

Fine-tune a DistilBERT Transformer to classify banking customer messages into
the 77 intents in [BANKING77](https://huggingface.co/datasets/PolyAI/banking77).
The training script downloads the dataset from Hugging Face the first time it
runs, uses a stratified validation split from the training data, and evaluates
the final model on the official test split.

## Requirements

- Python 3.10 or newer
- Internet access for the initial dataset and model downloads
- A CUDA-capable GPU is recommended; training also runs on CPU, but will be
  slower

## Setup

From this directory, create a virtual environment and install the dependencies
from the project requirements file:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Using the virtual environment's Python directly avoids PowerShell activation
policy issues and keeps the project's packages separate from other Python
installations.

## Export dataset to CSV

To download BANKING77 and save the official train and test splits as CSV files:

```powershell
.\.venv\Scripts\python.exe export_dataset.py
```

This downloads the official source CSVs from the BANKING77 dataset repository
and saves `data/banking77_train.csv` (10,003 rows) and
`data/banking77_test.csv` (3,080 rows). Each CSV has `text` and `intent`
columns, with human-readable intent names. Internet access is required.

## Train and evaluate

```powershell
.\.venv\Scripts\python.exe train.py
```

By default, this fine-tunes `distilbert-base-uncased` for three epochs, saves
the best validation-accuracy checkpoint to `outputs/banking77-distilbert`, and
reports accuracy and macro F1 on the official test set. The output directory
contains the model, tokenizer, label names, and `test_results.json`.

Useful options:

```powershell
.\.venv\Scripts\python.exe train.py --epochs 5 --train-batch-size 8 --max-length 160
.\.venv\Scripts\python.exe train.py --model-name bert-base-uncased --output-dir outputs/banking77-bert
```

Training uses the generated CSV files. Training checkpoints are selected using
a stratified validation split—not the official test set—to avoid using test
data for model selection.

## Predict intents

After training, pass one or more messages with repeated `--text` options:

```powershell
.\.venv\Scripts\python.exe predict.py --text "I tried to withdraw money but the ATM did not give me cash."
.\.venv\Scripts\python.exe predict.py --text "My card was swallowed by the ATM." --text "I need to report a fraudulent transaction."
```

The script prints one JSON object per message with the top three predicted
intents and their confidence scores. Set `--top-k` to change the number of
suggestions:

```powershell
.\.venv\Scripts\python.exe predict.py --top-k 1 --text "I have not received my new card."
```

To use a different trained model directory, pass `--model-dir`.

## Streamlit user interface

Start the browser UI after training:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit opens the app in your browser. Enter a customer message and select
**Classify intent** to see the most likely intent, confidence, and ranked
alternatives. The app includes example messages, model/test-metric status, a
downloadable JSON prediction, and a sidebar option for selecting another
trained model folder. The **Batch classification** tab also accepts pasted
messages or a CSV with a `text` column (up to 500 messages), flags low-confidence
predictions for review, and downloads results as CSV. The default model folder
is `outputs/banking77-distilbert`.
