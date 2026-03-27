from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_PATH = Path("results") / "results.csv"
OUTPUT_DIR = Path("results")
F1_CHART_PATH = OUTPUT_DIR / "chart_f1_comparison.png"
ACCURACY_LATENCY_CHART_PATH = OUTPUT_DIR / "chart_accuracy_vs_latency.png"
ALL_METRICS_CHART_PATH = OUTPUT_DIR / "chart_all_metrics.png"


def load_results(results_path: Path) -> pd.DataFrame:
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    df = pd.read_csv(results_path)
    if df.empty:
        raise ValueError("results/results.csv is empty. Run benchmark.py first.")

    return df


def create_f1_chart(df: pd.DataFrame) -> None:
    colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F"]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(df["model"], df["f1"], color=colors[: len(df)])
    plt.title("F1 Score Comparison — Sentiment140 Benchmark")
    plt.xlabel("Model")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1)

    for bar, value in zip(bars, df["f1"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(F1_CHART_PATH, dpi=300)
    plt.close()


def create_accuracy_vs_latency_chart(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(df["latency_seconds"], df["accuracy"], s=100, c="#4E79A7")
    plt.title("Accuracy vs Latency Tradeoff")
    plt.xlabel("Latency (seconds)")
    plt.ylabel("Accuracy")

    for _, row in df.iterrows():
        plt.annotate(
            row["model"],
            (row["latency_seconds"], row["accuracy"]),
            textcoords="offset points",
            xytext=(6, 6),
        )

    plt.tight_layout()
    plt.savefig(ACCURACY_LATENCY_CHART_PATH, dpi=300)
    plt.close()


def create_all_metrics_chart(df: pd.DataFrame) -> None:
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(df))
    width = 0.2
    colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2"]

    plt.figure(figsize=(12, 7))
    for index, metric in enumerate(metrics):
        plt.bar(
            x + (index - 1.5) * width,
            df[metric],
            width=width,
            label=metric.capitalize(),
            color=colors[index],
        )

    plt.title("All Metrics by Model")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.xticks(x, df["model"])
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ALL_METRICS_CHART_PATH, dpi=300)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df = load_results(RESULTS_PATH)
    create_f1_chart(results_df)
    create_accuracy_vs_latency_chart(results_df)
    create_all_metrics_chart(results_df)
    print("Charts saved to results/")


if __name__ == "__main__":
    main()
