"""Streamlit interface for BANKING77 intent classification."""

import csv
import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from classifier import load_classifier, predict_intents


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJECT_DIR / "outputs" / "banking77-distilbert"
MAX_BATCH_MESSAGES = 500
EXAMPLE_MESSAGES = {
    "ATM did not dispense cash": (
        "I tried to withdraw money but the ATM did not give me cash."
    ),
    "Card has not arrived": "I ordered my card over a week ago and it still has not arrived.",
    "Transfer not received": "I sent a bank transfer yesterday but the recipient has not received it.",
    "Card payment declined": "My card payment was declined even though I have enough money.",
}


st.set_page_config(
    page_title="Banking Intent Classifier",
    page_icon="🏦",
    layout="wide",
)
st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2.2rem; padding-bottom: 3rem;}
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse at 10% 0%, rgba(32, 112, 140, .13), transparent 38%),
            var(--background-color);
    }
    .hero {
        padding: 1.55rem 1.8rem;
        border: 1px solid rgba(120, 160, 180, .22);
        border-radius: 18px;
        background: linear-gradient(115deg, rgba(17, 68, 84, .84), rgba(30, 47, 74, .80));
        margin-bottom: 1.5rem;
    }
    .hero-kicker {
        color: #9ee6d0;
        text-transform: uppercase;
        letter-spacing: .13em;
        font-size: .75rem;
        font-weight: 700;
        margin-bottom: .45rem;
    }
    .hero h1 {padding: 0; margin: 0; color: #fff; font-size: 2rem;}
    .hero p {color: #d3e1eb; margin: .55rem 0 0; font-size: 1rem;}
    .section-label {
        color: #7dcdb9;
        text-transform: uppercase;
        letter-spacing: .11em;
        font-size: .75rem;
        font-weight: 700;
        margin-bottom: .35rem;
    }
    [data-testid="stMetric"] {
        padding: .85rem 1rem;
        border: 1px solid rgba(120, 160, 180, .2);
        border-radius: 12px;
        background: rgba(90, 130, 150, .08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_classifier(model_dir: str):
    return load_classifier(model_dir)


def use_example():
    st.session_state.customer_message = EXAMPLE_MESSAGES[
        st.session_state.example_choice
    ]


def render_single_result(result):
    st.divider()
    st.markdown(
        '<div class="section-label">Classification result</div>',
        unsafe_allow_html=True,
    )
    primary_prediction = result["predictions"][0]
    with st.container(border=True):
        result_columns = st.columns([3, 1])
        with result_columns[0]:
            st.caption("Most likely intent")
            st.subheader(primary_prediction["intent"].title())
            st.caption("Based on the message you entered")
        with result_columns[1]:
            st.metric("Confidence", f"{primary_prediction['confidence']:.1%}")

        alternatives = result["predictions"][1:]
        if alternatives:
            st.markdown("**Other likely intents**")
            for prediction in alternatives:
                st.progress(
                    prediction["confidence"],
                    text=(
                        f"{prediction['intent'].title()} · "
                        f"{prediction['confidence']:.1%}"
                    ),
                )

    st.download_button(
        "Download result as JSON",
        data=json.dumps(
            {"text": result["message"], "predictions": result["predictions"]},
            ensure_ascii=False,
            indent=2,
        ),
        file_name="intent_prediction.json",
        mime="application/json",
        key="download_single_result",
    )


def parse_uploaded_csv(uploaded_file):
    try:
        content = uploaded_file.getvalue().decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Upload a UTF-8 encoded CSV file.") from error

    reader = csv.DictReader(io.StringIO(content))
    if not reader.fieldnames or "text" not in reader.fieldnames:
        raise ValueError("The CSV must contain a column named `text`.")

    messages = [
        row.get("text", "").strip()
        for row in reader
        if row.get("text") and row["text"].strip()
    ]
    if not messages:
        raise ValueError("The CSV contains no non-empty messages in its `text` column.")
    return messages


st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">BANKING77 · AI-powered support</div>
      <h1>Banking Intent Classifier</h1>
      <p>Turn customer messages into clear, ranked support intents in seconds.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

model_dir = st.sidebar.text_input(
    "Trained model folder",
    value=str(DEFAULT_MODEL_DIR),
    help="The folder created by train.py after training completes.",
)
top_k = st.sidebar.slider("Intent suggestions", min_value=1, max_value=10, value=3)
review_threshold = st.sidebar.slider(
    "Flag confidence below",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Batch results below this confidence are marked for human review.",
)
model_path = Path(model_dir).expanduser()
has_trained_model = (model_path / "config.json").is_file()

st.sidebar.divider()
st.sidebar.subheader("Project status")
if has_trained_model:
    st.sidebar.success("Model ready")
    metrics_path = model_path / "test_results.json"
    if metrics_path.is_file():
        with metrics_path.open(encoding="utf-8") as metrics_file:
            test_metrics = json.load(metrics_file)
        metric_columns = st.sidebar.columns(2)
        metric_columns[0].metric(
            "Test accuracy", f"{test_metrics['test_accuracy']:.1%}"
        )
        metric_columns[1].metric(
            "Macro F1", f"{test_metrics['test_macro_f1']:.1%}"
        )
else:
    st.sidebar.warning("Model needs training")
    st.sidebar.caption("The BANKING77 CSVs are used to train the classifier.")
st.sidebar.caption("77 intents · English · DistilBERT")

if not has_trained_model:
    train_csv = PROJECT_DIR / "data" / "banking77_train.csv"
    test_csv = PROJECT_DIR / "data" / "banking77_test.csv"
    if train_csv.is_file() and test_csv.is_file():
        st.info(
            "Your BANKING77 CSV files are ready. Train the model from the "
            "project folder, then return here:\n\n"
            "```powershell\n.\\.venv\\Scripts\\python.exe train.py\n```"
        )
    else:
        st.info(
            "The model and dataset CSVs are missing. Run "
            "`.\\.venv\\Scripts\\python.exe export_dataset.py` to create the "
            "CSVs, then run `train.py`."
        )

single_tab, batch_tab = st.tabs(["Single message", "Batch classification"])

with single_tab:
    st.markdown(
        '<div class="section-label">Classify a message</div>',
        unsafe_allow_html=True,
    )
    st.write("Choose an example or enter a customer message in your own words.")

    example_columns = st.columns([3, 1])
    example_columns[0].selectbox(
        "Try a sample message",
        options=list(EXAMPLE_MESSAGES),
        key="example_choice",
        label_visibility="collapsed",
    )
    example_columns[1].button(
        "Use example",
        on_click=use_example,
        disabled=not has_trained_model,
        use_container_width=True,
    )

    with st.form("intent_classifier_form"):
        message = st.text_area(
            "Customer message",
            key="customer_message",
            placeholder="For example: I tried to withdraw money but the ATM did not give me cash.",
            height=145,
            max_chars=2000,
            help="Describe the customer's issue in a sentence or two.",
        )
        submitted = st.form_submit_button(
            "Classify intent",
            type="primary",
            use_container_width=True,
            disabled=not has_trained_model,
        )

    if submitted:
        if not message.strip():
            st.warning("Enter a customer message before classifying.")
        elif not has_trained_model:
            st.error(
                "No trained model was found. Run "
                "`.\\.venv\\Scripts\\python.exe train.py` from the project folder."
            )
        else:
            resolved_model_dir = str(model_path.resolve())
            try:
                with st.spinner("Reading the message and ranking likely intents..."):
                    tokenizer, model, device = get_classifier(resolved_model_dir)
                    predictions = predict_intents(
                        [message],
                        tokenizer,
                        model,
                        device,
                        top_k=top_k,
                    )[0]
            except (OSError, ValueError, RuntimeError) as error:
                st.error(f"Could not load the model or classify the message: {error}")
            else:
                st.session_state.last_result = {
                    "message": message,
                    "model_dir": resolved_model_dir,
                    "predictions": predictions,
                }

    single_result = st.session_state.get("last_result")
    if (
        single_result
        and single_result["message"] == message
        and single_result["model_dir"] == str(model_path.resolve())
    ):
        render_single_result(single_result)

with batch_tab:
    st.markdown(
        '<div class="section-label">Classify a support queue</div>',
        unsafe_allow_html=True,
    )
    st.write(
        "Paste one message per line or upload a CSV with a `text` column. "
        f"Batch size is limited to {MAX_BATCH_MESSAGES} messages."
    )
    input_method = st.radio(
        "Batch input",
        ["Paste messages", "Upload CSV"],
        horizontal=True,
        key="batch_input_method",
    )
    uploaded_file = None
    pasted_messages = ""
    if input_method == "Paste messages":
        with st.form("batch_text_form"):
            pasted_messages = st.text_area(
                "Customer messages",
                placeholder="I cannot use my card.\nMy transfer has not arrived.\nThe ATM kept my card.",
                height=180,
                help="Enter one complete customer message per line.",
            )
            batch_submitted = st.form_submit_button(
                "Classify messages",
                type="primary",
                use_container_width=True,
                disabled=not has_trained_model,
            )
    else:
        uploaded_file = st.file_uploader(
            "Upload a CSV file",
            type=["csv"],
            help="Include a header named `text`; other columns are ignored.",
            key="batch_csv_upload",
        )
        batch_submitted = st.button(
            "Classify uploaded CSV",
            type="primary",
            use_container_width=True,
            disabled=not has_trained_model or uploaded_file is None,
        )

    if batch_submitted:
        try:
            if uploaded_file is not None:
                batch_messages = parse_uploaded_csv(uploaded_file)
            else:
                batch_messages = [
                    line.strip()
                    for line in pasted_messages.splitlines()
                    if line.strip()
                ]
            if not batch_messages:
                raise ValueError("Add at least one non-empty message.")
            if len(batch_messages) > MAX_BATCH_MESSAGES:
                raise ValueError(
                    f"Batch has {len(batch_messages)} messages; the limit is "
                    f"{MAX_BATCH_MESSAGES}."
                )

            resolved_model_dir = str(model_path.resolve())
            with st.spinner(f"Classifying {len(batch_messages)} messages..."):
                tokenizer, model, device = get_classifier(resolved_model_dir)
                batch_predictions = predict_intents(
                    batch_messages,
                    tokenizer,
                    model,
                    device,
                    top_k=top_k,
                )
        except (OSError, ValueError, RuntimeError) as error:
            st.error(f"Could not classify this batch: {error}")
        else:
            st.session_state.batch_result = {
                "model_dir": resolved_model_dir,
                "rows": [
                    {"text": text, "predictions": predictions}
                    for text, predictions in zip(batch_messages, batch_predictions)
                ],
            }

    batch_result = st.session_state.get("batch_result")
    if batch_result and batch_result["model_dir"] == str(model_path.resolve()):
        rows = batch_result["rows"]
        results_frame = pd.DataFrame(
            [
                {
                    "Message": row["text"],
                    "Predicted intent": row["predictions"][0]["intent"],
                    "Confidence": row["predictions"][0]["confidence"],
                    "Review": (
                        "Review"
                        if row["predictions"][0]["confidence"] < review_threshold
                        else "Ready"
                    ),
                }
                for row in rows
            ]
        )
        review_count = int((results_frame["Review"] == "Review").sum())
        metric_columns = st.columns(3)
        metric_columns[0].metric("Messages classified", len(rows))
        metric_columns[1].metric("Ready", len(rows) - review_count)
        metric_columns[2].metric("Flagged for review", review_count)
        st.dataframe(
            results_frame,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    min_value=0.0,
                    max_value=1.0,
                    format="percent",
                )
            },
        )
        st.download_button(
            "Download batch results as CSV",
            data=results_frame.to_csv(index=False).encode("utf-8-sig"),
            file_name="banking77_predictions.csv",
            mime="text/csv",
            key="download_batch_results",
        )

st.caption(
    "Predictions are model suggestions, not decisions about a customer's "
    "account. Review messages with low confidence."
)
