import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from time import perf_counter
from urllib.request import urlretrieve
import zipfile

import nltk
import pandas as pd
import torch
from groq import Groq
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import pipeline

try:
    from google import genai as google_genai
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    try:
        import google.generativeai as genai_legacy
        GOOGLE_GENAI_AVAILABLE = False
    except ImportError:
        genai_legacy = None
        GOOGLE_GENAI_AVAILABLE = False

nltk.download("vader_lexicon", quiet=True)

DATASET_URL = "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
COLUMN_NAMES = ["label", "id", "date", "query", "user", "text"]
RAW_OUTPUT_PATH = Path("data") / "sentiment140_raw.csv"
SAMPLE_OUTPUT_PATH = Path("data") / "sentiment140_sample.csv"
VADER_RESULTS_PATH = Path("results") / "vader_results.csv"
METRICS_OUTPUT_PATH = Path("results") / "results.csv"
DISTILBERT_RESULTS_PATH = Path("results") / "distilbert_results.csv"
ROBERTA_RESULTS_PATH = Path("results") / "roberta_results.csv"
LLAMA_RESULTS_PATH = Path("results") / "llama_results.csv"
GEMINI_RESULTS_PATH = Path("results") / "gemini_results.csv"
SAMPLE_SIZE_PER_CLASS = 5000
EXPECTED_SAMPLE_SIZE = SAMPLE_SIZE_PER_CLASS * 2
RANDOM_STATE = 42
HF_TOKEN = os.environ.get("HF_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")


# ---------------------------------------------------------------------------
# Helper: check if a results file is already complete
# ---------------------------------------------------------------------------

def _results_complete(path: Path, expected_rows: int = EXPECTED_SAMPLE_SIZE) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
        return len(df) >= expected_rows
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Layer 1 — Data download and sampling
# ---------------------------------------------------------------------------

def download_and_clean_sentiment140(raw_output_path: Path) -> pd.DataFrame:
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=raw_output_path.parent) as temp_dir:
        zip_path = Path(temp_dir) / "trainingandtestdata.zip"
        urlretrieve(DATASET_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as archive:
            csv_name = next(
                (n for n in archive.namelist()
                 if n.endswith(".csv") and "training" in Path(n).name.lower()),
                None,
            )
            if csv_name is None:
                raise FileNotFoundError("Could not find training CSV inside Sentiment140 zip.")
            with archive.open(csv_name) as csv_file:
                df = pd.read_csv(csv_file, header=None, names=COLUMN_NAMES, encoding="latin-1")

    cleaned_df = (
        df.loc[:, ["label", "text"]]
        .loc[lambda f: f["label"].isin([0, 4])]
        .assign(label=lambda f: f["label"].map({0: 0, 4: 1}).astype("int64"))
    )
    cleaned_df.to_csv(raw_output_path, index=False)
    return cleaned_df


def load_or_create_raw_data(raw_output_path: Path) -> pd.DataFrame:
    if raw_output_path.exists():
        print(f"Raw dataset already exists at {raw_output_path}. Skipping download.")
        return pd.read_csv(raw_output_path)
    print("Downloading and cleaning Sentiment140 dataset...")
    return download_and_clean_sentiment140(raw_output_path)


def create_stratified_sample(df: pd.DataFrame, sample_output_path: Path) -> pd.DataFrame:
    counts = df["label"].value_counts()
    for label in [0, 1]:
        if counts.get(label, 0) < SAMPLE_SIZE_PER_CLASS:
            raise ValueError(f"Not enough rows for label {label}.")
    sample_df = (
        df.groupby("label", group_keys=False)
        .sample(n=SAMPLE_SIZE_PER_CLASS, random_state=RANDOM_STATE)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    sample_df.to_csv(sample_output_path, index=False)
    return sample_df


def load_or_create_sample(df: pd.DataFrame, sample_output_path: Path) -> pd.DataFrame:
    if _results_complete(sample_output_path, EXPECTED_SAMPLE_SIZE):
        print(f"Sample already exists at {sample_output_path}. Skipping sampling.")
        return pd.read_csv(sample_output_path)
    print("Creating stratified 10,000-row sample...")
    return create_stratified_sample(df, sample_output_path)


def print_data_summary(raw_df: pd.DataFrame, sample_df: pd.DataFrame) -> None:
    print("\nSummary")
    print(f"Raw rows: {len(raw_df):,}")
    print(f"Raw label distribution: {raw_df['label'].value_counts().sort_index().to_dict()}")
    print(f"Sample rows: {len(sample_df):,}")
    print(f"Sample label distribution: {sample_df['label'].value_counts().sort_index().to_dict()}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def append_metrics_rows(metrics_rows: list[dict], metrics_output_path: Path) -> None:
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    if metrics_output_path.exists():
        existing = pd.read_csv(metrics_output_path)
        existing = existing.loc[~existing["model"].isin([r["model"] for r in metrics_rows])]
        metrics_df = pd.concat([existing, pd.DataFrame(metrics_rows)], ignore_index=True)
    else:
        metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(metrics_output_path, index=False)


def print_model_summary(
    model_name: str, accuracy: float, precision: float,
    recall: float, f1: float, latency_seconds: float, cost_per_1k: float,
) -> None:
    print(f"\n=== {model_name} Results ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Latency:   {latency_seconds:.2f} seconds")
    print(f"Cost/1K:   ${cost_per_1k:.2f}")


# ---------------------------------------------------------------------------
# Layer 2 — VADER
# ---------------------------------------------------------------------------

def run_vader(
    sample_input_path: Path,
    vader_results_path: Path = VADER_RESULTS_PATH,
    metrics_output_path: Path = METRICS_OUTPUT_PATH,
) -> None:
    if _results_complete(vader_results_path):
        print("\nVADER results already exist. Skipping.")
        return

    vader_results_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df = pd.read_csv(sample_input_path)
    analyzer = SentimentIntensityAnalyzer()

    start_time = perf_counter()
    compound_scores = sample_df["text"].fillna("").astype(str).apply(
        lambda t: analyzer.polarity_scores(t)["compound"]
    )
    latency_seconds = perf_counter() - start_time

    predicted_labels = (compound_scores >= 0.05).astype(int)
    true_labels = sample_df["label"].astype(int)

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    pd.DataFrame({
        "text": sample_df["text"],
        "true_label": true_labels,
        "predicted_label": predicted_labels,
        "compound_score": compound_scores,
    }).to_csv(vader_results_path, index=False)

    pd.DataFrame([{
        "model": "VADER", "accuracy": round(accuracy, 4),
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "latency_seconds": round(latency_seconds, 4),
        "cost_per_1k": 0.0,
    }]).to_csv(metrics_output_path, index=False)

    print_model_summary("VADER", accuracy, precision, recall, f1, latency_seconds, 0.0)


# ---------------------------------------------------------------------------
# Layer 3 — HuggingFace Transformers (distilBERT + RoBERTa)
# ---------------------------------------------------------------------------

def run_single_transformer_model(
    model_name: str, model_id: str, sample_df: pd.DataFrame,
    results_path: Path, label_map: dict, metrics_output_path: Path, device: int,
) -> None:
    if _results_complete(results_path):
        print(f"\n{model_name} results already exist. Skipping.")
        return

    classifier = pipeline(
        "text-classification", model=model_id, tokenizer=model_id,
        device=device, token=HF_TOKEN,
    )
    texts = sample_df["text"].fillna("").astype(str).tolist()
    true_labels = sample_df["label"].astype(int)

    start_time = perf_counter()
    predictions = classifier(texts, batch_size=32, truncation=True, max_length=512)
    latency_seconds = perf_counter() - start_time

    prediction_df = pd.DataFrame(predictions)
    prediction_df["predicted_label"] = prediction_df["label"].map(label_map)
    if prediction_df["predicted_label"].isna().any():
        unexpected = sorted(prediction_df.loc[prediction_df["predicted_label"].isna(), "label"].unique())
        raise ValueError(f"Unexpected labels from {model_name}: {unexpected}")

    predicted_labels = prediction_df["predicted_label"].astype(int)
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    pd.DataFrame({
        "text": sample_df["text"], "true_label": true_labels,
        "predicted_label": predicted_labels, "score": prediction_df["score"],
    }).to_csv(results_path, index=False)

    append_metrics_rows([{
        "model": model_name, "accuracy": round(accuracy, 4),
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "latency_seconds": round(latency_seconds, 4),
        "cost_per_1k": 0.0,
    }], metrics_output_path)
    print_model_summary(model_name, accuracy, precision, recall, f1, latency_seconds, 0.0)


def run_transformers(
    sample_input_path: Path,
    metrics_output_path: Path = METRICS_OUTPUT_PATH,
) -> None:
    sample_df = pd.read_csv(sample_input_path)
    device = 0 if torch.cuda.is_available() else -1

    run_single_transformer_model(
        "distilBERT", "distilbert-base-uncased-finetuned-sst-2-english",
        sample_df, DISTILBERT_RESULTS_PATH,
        {"NEGATIVE": 0, "POSITIVE": 1}, metrics_output_path, device,
    )
    run_single_transformer_model(
        "RoBERTa", "cardiffnlp/twitter-roberta-base-sentiment",
        sample_df, ROBERTA_RESULTS_PATH,
        {"LABEL_0": 0, "LABEL_1": 0, "LABEL_2": 1}, metrics_output_path, device,
    )


# ---------------------------------------------------------------------------
# Layer 4 — LLaMA 3.1 8B via Groq
# ---------------------------------------------------------------------------

def run_llama(
    sample_input_path: Path,
    llama_results_path: Path = LLAMA_RESULTS_PATH,
    metrics_output_path: Path = METRICS_OUTPUT_PATH,
) -> None:
    if _results_complete(llama_results_path, expected_rows=500):
        print("\nLLaMA results already exist. Skipping.")
        return

    if not GROQ_API_KEY:
        print("\nWarning: GROQ_API_KEY is not set. Skipping LLaMA benchmark.")
        return

    llama_results_path.parent.mkdir(parents=True, exist_ok=True)
    #sample_df = pd.read_csv(sample_input_path)
    sample_df = pd.read_csv(sample_input_path).sample(n=500, random_state=42).reset_index(drop=True)
    texts = sample_df["text"].fillna("").astype(str).tolist()
    true_labels = sample_df["label"].astype(int)

    client = Groq(api_key=GROQ_API_KEY)
    raw_responses: list[str] = []
    predicted_labels: list[int] = []

    print("\nRunning LLaMA 3.1 8B via Groq...")
    start_time = perf_counter()
    for batch_start in range(0, len(texts), 10):
        batch = texts[batch_start: batch_start + 10]
        for text in batch:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a sentiment classifier. Reply with exactly one word: positive or negative."},
                    {"role": "user", "content": f"Tweet: {text}\nSentiment:"},
                ],
            )
            raw = response.choices[0].message.content.strip()
            raw_responses.append(raw)
            predicted_labels.append(1 if "positive" in raw.lower() else 0)

        if batch_start + 10 < len(texts):
            time.sleep(1)

        if (batch_start // 10 + 1) % 100 == 0:
            print(f"  LLaMA: {batch_start + 10}/{len(texts)} tweets processed...")

    latency_seconds = perf_counter() - start_time
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    pd.DataFrame({
        "text": sample_df["text"], "true_label": true_labels,
        "predicted_label": predicted_labels, "raw_response": raw_responses,
    }).to_csv(llama_results_path, index=False)

    append_metrics_rows([{
        "model": "LLaMA", "accuracy": round(accuracy, 4),
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "latency_seconds": round(latency_seconds, 4),
        "cost_per_1k": 0.0,
    }], metrics_output_path)
    print_model_summary("LLaMA", accuracy, precision, recall, f1, latency_seconds, 0.0)


# ---------------------------------------------------------------------------
# Layer 5 — Gemini 1.5 Flash
# ---------------------------------------------------------------------------

def run_gemini(
    sample_input_path: Path,
    gemini_results_path: Path = GEMINI_RESULTS_PATH,
    metrics_output_path: Path = METRICS_OUTPUT_PATH,
) -> None:
    if _results_complete(gemini_results_path, expected_rows=500):
        print("\nGemini results already exist. Skipping.")
        return

    if not GOOGLE_API_KEY:
        print("\nWarning: GOOGLE_API_KEY is not set. Skipping Gemini benchmark.")
        return

    gemini_results_path.parent.mkdir(parents=True, exist_ok=True)
    #sample_df = pd.read_csv(sample_input_path)
    sample_df = pd.read_csv(sample_input_path).sample(n=500, random_state=42).reset_index(drop=True)
    texts = sample_df["text"].fillna("").astype(str).tolist()
    true_labels = sample_df["label"].astype(int)

    # Use new google.genai if available, fall back to legacy- used openrouter when gemini hit dailylimit
    if GOOGLE_GENAI_AVAILABLE:
        import openai
        client_or = openai.OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        def get_response(prompt):
            r = client_or.chat.completions.create(
                model="google/gemini-2.0-flash-001",
                messages=[{"role": "user", "content": prompt}]
            )
            return (r.choices[0].message.content or "").strip()
    """if GOOGLE_GENAI_AVAILABLE:
        client = google_genai.Client(api_key=GOOGLE_API_KEY)
        def get_response(prompt):
            r = client.models.generate_content(model="gemini-1.5-flash-001", contents=prompt)
            return (r.text or "").strip()
    else:
        genai_legacy.configure(api_key=GOOGLE_API_KEY)
        model = genai_legacy.GenerativeModel("gemini-2.0-flash")
        def get_response(prompt):
            r = model.generate_content(prompt)
            return (r.text or "").strip()"""

    raw_responses: list[str] = []
    predicted_labels: list[int] = []

    print("\nRunning Gemini 1.5 Flash...")
    start_time = perf_counter()
    for batch_start in range(0, len(texts), 10):
        batch = texts[batch_start: batch_start + 10]
        for text in batch:
            prompt = (
                f"Classify the sentiment of this tweet as exactly one word — positive or negative.\n"
                f"Tweet: {text}\nSentiment:"
            )
            raw = get_response(prompt)
            raw_responses.append(raw)
            predicted_labels.append(1 if "positive" in raw.lower() else 0)

        if batch_start + 10 < len(texts):
            time.sleep(2)

        if (batch_start // 10 + 1) % 100 == 0:
            print(f"  Gemini: {batch_start + 10}/{len(texts)} tweets processed...")

    latency_seconds = perf_counter() - start_time
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    pd.DataFrame({
        "text": sample_df["text"], "true_label": true_labels,
        "predicted_label": predicted_labels, "raw_response": raw_responses,
    }).to_csv(gemini_results_path, index=False)

    append_metrics_rows([{
        "model": "Gemini", "accuracy": round(accuracy, 4),
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "latency_seconds": round(latency_seconds, 4),
        "cost_per_1k": 0.0,
    }], metrics_output_path)
    print_model_summary("Gemini", accuracy, precision, recall, f1, latency_seconds, 0.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    raw_df = load_or_create_raw_data(RAW_OUTPUT_PATH)
    sample_df = load_or_create_sample(raw_df, SAMPLE_OUTPUT_PATH)
    print_data_summary(raw_df, sample_df)
    run_vader(SAMPLE_OUTPUT_PATH)
    run_transformers(SAMPLE_OUTPUT_PATH)
    run_llama(SAMPLE_OUTPUT_PATH)
    run_gemini(SAMPLE_OUTPUT_PATH)


if __name__ == "__main__":
    main()