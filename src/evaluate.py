"""
Evaluation and Visualization Module.
Computes quantitative measures (Loss, Perplexity, Accuracy, Top-5 Accuracy, Training Time, Parameters)
and generates publication-quality visualization figures.
"""

import time
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from src.config import RESULTS_DIR

# Set clean aesthetic plot style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11


def compute_perplexity(loss_value: float) -> float:
    """Compute perplexity: PPL = exp(loss). Clipped to prevent overflow."""
    return float(np.exp(np.clip(loss_value, -10.0, 15.0)))


def measure_inference_speed(model: tf.keras.Model, seq_length: int = 20, num_trials: int = 100) -> float:
    """Measure inference latency in milliseconds per predicted token."""
    dummy_input = np.random.randint(1, 1000, size=(1, seq_length), dtype=np.int32)
    # Warmup
    for _ in range(10):
        _ = model(dummy_input, training=False)

    t0 = time.perf_counter()
    for _ in range(num_trials):
        _ = model(dummy_input, training=False)
    t1 = time.perf_counter()

    latency_ms = ((t1 - t0) / num_trials) * 1000.0
    return round(latency_ms, 2)


def generate_quantitative_summary(
    models_dict: Dict[str, tf.keras.Model],
    histories_dict: Dict[str, dict],
    training_times: Dict[str, float]
) -> pd.DataFrame:
    """
    Compile a structured DataFrame comparing all models across all quantitative metrics.
    """
    records = []
    for name, model in models_dict.items():
        hist = histories_dict[name]
        final_train_loss = hist["loss"][-1]
        final_val_loss = hist["val_loss"][-1]
        final_train_acc = hist["accuracy"][-1]
        final_val_acc = hist["val_accuracy"][-1]
        final_train_top5 = hist.get("top_5_accuracy", [0])[-1]
        final_val_top5 = hist.get("val_top_5_accuracy", [0])[-1]

        train_ppl = compute_perplexity(final_train_loss)
        val_ppl = compute_perplexity(final_val_loss)
        best_val_loss = min(hist["val_loss"])
        best_val_ppl = compute_perplexity(best_val_loss)

        total_params = model.count_params()
        train_time = training_times[name]
        inf_latency = measure_inference_speed(model)

        records.append({
            "Model": name.replace("_", " "),
            "Parameters": total_params,
            "Train Loss": round(final_train_loss, 4),
            "Val Loss": round(final_val_loss, 4),
            "Best Val Loss": round(best_val_loss, 4),
            "Train PPL": round(train_ppl, 2),
            "Val PPL": round(val_ppl, 2),
            "Best Val PPL": round(best_val_ppl, 2),
            "Train Acc (%)": round(final_train_acc * 100, 2),
            "Val Acc (%)": round(final_val_acc * 100, 2),
            "Val Top-5 Acc (%)": round(final_val_top5 * 100, 2),
            "Train Time (s)": round(train_time, 2),
            "Inference (ms/token)": inf_latency
        })

    df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / "quantitative_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Quantitative metrics saved to {csv_path}")
    return df


def plot_training_curves(histories_dict: Dict[str, dict], save_dir: Path = RESULTS_DIR):
    """
    Generate 4-panel comprehensive training trajectory figure:
    1. Training & Validation Loss
    2. Validation Perplexity
    3. Validation Accuracy (Top-1)
    4. Validation Top-5 Accuracy
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(next(iter(histories_dict.values()))["loss"]) + 1)
    palette = {"Simple_RNN": "#e74c3c", "Bidirectional_RNN": "#e67e22", "LSTM": "#2ecc71", "GRU": "#3498db"}

    # 1. Validation Loss
    ax1 = axes[0, 0]
    for name, hist in histories_dict.items():
        label_clean = name.replace("_", " ")
        ax1.plot(epochs, hist["val_loss"], label=f"{label_clean} (Val)", color=palette.get(name, "black"), lw=2.2)
        ax1.plot(epochs, hist["loss"], linestyle="--", color=palette.get(name, "black"), alpha=0.5)
    ax1.set_title("(a) Cross-Entropy Loss (Solid: Val, Dashed: Train)", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, alpha=0.4)

    # 2. Validation Perplexity
    ax2 = axes[0, 1]
    for name, hist in histories_dict.items():
        val_ppl = [compute_perplexity(v) for v in hist["val_loss"]]
        ax2.plot(epochs, val_ppl, label=name.replace("_", " "), color=palette.get(name, "black"), lw=2.2)
    ax2.set_title("(b) Validation Perplexity (PPL = exp(Val Loss))", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Perplexity")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, alpha=0.4)

    # 3. Validation Accuracy
    ax3 = axes[1, 0]
    for name, hist in histories_dict.items():
        val_acc = [v * 100 for v in hist["val_accuracy"]]
        ax3.plot(epochs, val_acc, label=name.replace("_", " "), color=palette.get(name, "black"), lw=2.2)
    ax3.set_title("(c) Validation Top-1 Accuracy (%)", fontweight="bold")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Top-1 Accuracy (%)")
    ax3.legend(loc="lower right", frameon=True)
    ax3.grid(True, alpha=0.4)

    # 4. Validation Top-5 Accuracy
    ax4 = axes[1, 1]
    for name, hist in histories_dict.items():
        if "val_top_5_accuracy" in hist:
            val_top5 = [v * 100 for v in hist["val_top_5_accuracy"]]
            ax4.plot(epochs, val_top5, label=name.replace("_", " "), color=palette.get(name, "black"), lw=2.2)
    ax4.set_title("(d) Validation Top-5 Accuracy (%)", fontweight="bold")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Top-5 Accuracy (%)")
    ax4.legend(loc="lower right", frameon=True)
    ax4.grid(True, alpha=0.4)

    plt.tight_layout()
    plot_path = save_dir / "training_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves plot to {plot_path}")


def plot_metric_comparisons(summary_df: pd.DataFrame, save_dir: Path = RESULTS_DIR):
    """
    Generate clean comparative bar charts for Perplexity, Accuracy, and Training Time.
    """
    colors = ["#e74c3c", "#e67e22", "#2ecc71", "#3498db"]

    # 1. Perplexity Comparison
    plt.figure(figsize=(8, 5))
    bars = plt.bar(summary_df["Model"], summary_df["Val PPL"], color=colors, edgecolor="black", alpha=0.85, width=0.55)
    plt.title("Validation Perplexity Comparison (Lower is Better)", fontweight="bold", pad=12)
    plt.ylabel("Validation Perplexity (PPL)")
    plt.grid(axis="y", alpha=0.4)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "perplexity_comparison.png", dpi=300)
    plt.close()

    # 2. Accuracy Comparison (Top-1 vs Top-5)
    plt.figure(figsize=(9, 5))
    x = np.arange(len(summary_df["Model"]))
    width = 0.35
    plt.bar(x - width / 2, summary_df["Val Acc (%)"], width, label="Top-1 Accuracy (%)", color="#3498db", edgecolor="black")
    plt.bar(x + width / 2, summary_df["Val Top-5 Acc (%)"], width, label="Top-5 Accuracy (%)", color="#2ecc71", edgecolor="black")
    plt.xticks(x, summary_df["Model"])
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy Comparison (Higher is Better)", fontweight="bold", pad=12)
    plt.legend(frameon=True)
    plt.grid(axis="y", alpha=0.4)
    for i in range(len(summary_df)):
        plt.text(x[i] - width / 2, summary_df["Val Acc (%)"][i] + 0.5, f"{summary_df['Val Acc (%)'][i]:.1f}%", ha="center", va="bottom", fontsize=9)
        plt.text(x[i] + width / 2, summary_df["Val Top-5 Acc (%)"][i] + 0.5, f"{summary_df['Val Top-5 Acc (%)'][i]:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_dir / "accuracy_comparison.png", dpi=300)
    plt.close()

    # 3. Training Time Comparison
    plt.figure(figsize=(8, 5))
    bars = plt.bar(summary_df["Model"], summary_df["Train Time (s)"], color=colors, edgecolor="black", alpha=0.85, width=0.55)
    plt.title("Total Training Time Comparison (15 Epochs)", fontweight="bold", pad=12)
    plt.ylabel("Wall-Clock Time (seconds)")
    plt.grid(axis="y", alpha=0.4)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.1f}s", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "training_time_comparison.png", dpi=300)
    plt.close()
    print("Saved all metric comparison bar charts.")


def plot_temperature_analysis(aggregated_metrics: List[dict], save_dir: Path = RESULTS_DIR):
    """
    Plot the effect of sampling temperature on Distinct-1 diversity and Trigram repetition.
    """
    df = pd.DataFrame(aggregated_metrics)
    csv_path = save_dir / "temperature_diversity_metrics.csv"
    df.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    palette = {"Simple_RNN": "#e74c3c", "Bidirectional_RNN": "#e67e22", "LSTM": "#2ecc71", "GRU": "#3498db"}

    # Left: Lexical Diversity (Distinct-1)
    ax1 = axes[0]
    for model_name in df["model"].unique():
        sub = df[df["model"] == model_name]
        ax1.plot(sub["temperature"], sub["avg_distinct_1"], marker="o", lw=2.2, label=model_name.replace("_", " "), color=palette.get(model_name, "black"))
    ax1.set_title("Lexical Diversity (Distinct-1) vs. Temperature", fontweight="bold")
    ax1.set_xlabel("Sampling Temperature (T)")
    ax1.set_ylabel("Distinct-1 Ratio (Higher = More Diverse)")
    ax1.set_xticks([0.2, 0.7, 1.2])
    ax1.legend(frameon=True)
    ax1.grid(True, alpha=0.4)

    # Right: Repetition Rate (Repetition-3)
    ax2 = axes[1]
    for model_name in df["model"].unique():
        sub = df[df["model"] == model_name]
        ax2.plot(sub["temperature"], sub["avg_repetition_3"] * 100, marker="s", lw=2.2, label=model_name.replace("_", " "), color=palette.get(model_name, "black"))
    ax2.set_title("Trigram Repetition Rate (%) vs. Temperature", fontweight="bold")
    ax2.set_xlabel("Sampling Temperature (T)")
    ax2.set_ylabel("Repeated 3-grams (%) (Lower = Less Repetitive)")
    ax2.set_xticks([0.2, 0.7, 1.2])
    ax2.legend(frameon=True)
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    plot_path = save_dir / "temperature_diversity_analysis.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved temperature diversity analysis plot to {plot_path}")
