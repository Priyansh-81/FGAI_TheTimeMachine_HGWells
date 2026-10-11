"""
End-to-End Orchestrator for Autoregressive Text Generation Experiment
Assignment 2 - Foundations of Generative AI (ICT4403)

Executes data preprocessing, trains all four models (Simple RNN, BiRNN, LSTM, GRU),
evaluates quantitative metrics, produces visual plots, conducts temperature-based
autoregressive text generation, and saves all artifacts for submission.
"""

import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import (
    BATCH_SIZE,
    EPOCHS,
    MODELS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    SAMPLES_DIR,
    SEED_SEQUENCES,
    TEMPERATURES,
    GENERATION_LENGTH,
)
from src.dataset import prepare_dataset
from src.evaluate import (
    generate_quantitative_summary,
    plot_metric_comparisons,
    plot_temperature_analysis,
    plot_training_curves,
)
from src.generator import generate_and_evaluate_samples
from src.models import (
    build_bidirectional_rnn,
    build_gru,
    build_lstm,
    build_simple_rnn,
)

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def main():
    print("=" * 80)
    print("FOUNDATIONS OF GENERATIVE AI (ICT4403) - ASSIGNMENT 2")
    print("AUTOREGRESSIVE TEXT GENERATION: SIMPLE RNN vs BiRNN vs LSTM vs GRU")
    print("=" * 80)

    # 1. Prepare Dataset
    print("\n[STEP 1/5] Preparing Dataset...")
    (X_train, y_train), (X_val, y_val), word2idx, idx2word, tokens = prepare_dataset()
    vocab_size = len(word2idx)
    print(f"Total vocabulary size: {vocab_size}")
    print(f"Training tensors: X_train shape={X_train.shape}, y_train shape={y_train.shape}")
    print(f"Validation tensors: X_val shape={X_val.shape}, y_val shape={y_val.shape}")

    # 2. Instantiate Models
    print("\n[STEP 2/5] Instantiating Architectures under Identical Experimental Conditions...")
    models_dict = {
        "Simple_RNN": build_simple_rnn(vocab_size=vocab_size),
        "Bidirectional_RNN": build_bidirectional_rnn(vocab_size=vocab_size),
        "LSTM": build_lstm(vocab_size=vocab_size),
        "GRU": build_gru(vocab_size=vocab_size),
    }

    for name, model in models_dict.items():
        print(f"\nModel: {name}")
        model.summary()

    # 3. Train Models
    print("\n[STEP 3/5] Training All Four Models...")
    histories_dict = {}
    training_times = {}

    history_records = []

    for name, model in models_dict.items():
        print("\n" + "-" * 60)
        print(f"Training {name.replace('_', ' ')} for {EPOCHS} epochs (Batch size: {BATCH_SIZE})...")
        print("-" * 60)

        t_start = time.perf_counter()
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1
        )
        t_elapsed = time.perf_counter() - t_start
        training_times[name] = t_elapsed
        histories_dict[name] = history.history

        # Save model weights / architecture
        model_save_path = MODELS_DIR / f"{name.lower()}.keras"
        model.save(model_save_path)
        print(f"--> Saved {name} to {model_save_path} (Training time: {t_elapsed:.2f}s)")

        # Record per-epoch history
        for ep in range(EPOCHS):
            history_records.append({
                "Model": name,
                "Epoch": ep + 1,
                "Train_Loss": history.history["loss"][ep],
                "Val_Loss": history.history["val_loss"][ep],
                "Train_Acc": history.history["accuracy"][ep],
                "Val_Acc": history.history["val_accuracy"][ep],
                "Val_Top5_Acc": history.history.get("val_top_5_accuracy", [0])[ep],
            })

    # Save detailed training history
    history_df = pd.DataFrame(history_records)
    history_df.to_csv(RESULTS_DIR / "training_history.csv", index=False)

    # 4. Quantitative Evaluation and Plotting
    print("\n[STEP 4/5] Computing Quantitative Metrics & Generating Comparison Plots...")
    summary_df = generate_quantitative_summary(models_dict, histories_dict, training_times)
    print("\n" + "=" * 80)
    print("QUANTITATIVE EVALUATION SUMMARY TABLE:")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    plot_training_curves(histories_dict, RESULTS_DIR)
    plot_metric_comparisons(summary_df, RESULTS_DIR)

    # 5. Temperature Sampling and Qualitative Analysis
    print("\n[STEP 5/5] Generating Autoregressive Text across Temperatures & Seeds...")
    detailed_samples, aggregated_metrics = generate_and_evaluate_samples(
        models_dict=models_dict,
        seed_sequences=SEED_SEQUENCES,
        word2idx=word2idx,
        idx2word=idx2word,
        temperatures=TEMPERATURES,
        num_words=GENERATION_LENGTH
    )

    plot_temperature_analysis(aggregated_metrics, RESULTS_DIR)

    # Save sample outputs
    json_path = SAMPLES_DIR / "generated_samples.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detailed_samples, f, indent=2)

    # Format human-readable markdown table of samples
    md_path = SAMPLES_DIR / "samples_by_temperature.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Qualitative Text Generation Samples across Models and Temperatures\n\n")
        f.write(f"**Dataset**: The Time Machine by H. G. Wells\n")
        f.write(f"**Prompt Seeds**: {len(SEED_SEQUENCES)} diverse seed sentences\n")
        f.write(f"**Sampling Temperatures**: {TEMPERATURES} (Low: 0.2, Moderate: 0.7, High: 1.2)\n\n")

        for seed in SEED_SEQUENCES:
            f.write(f"## Seed Prompt: *\"{seed}\"*\n\n")
            for model_name in models_dict.keys():
                f.write(f"### {model_name.replace('_', ' ')}\n\n")
                f.write("| Temperature | Generated Continuation | Distinct-1 | Distinct-2 | Rep-3 (%) |\n")
                f.write("|:---:|:---|:---:|:---:|:---:|\n")
                for s in detailed_samples:
                    if s["model"] == model_name and s["seed"] == seed:
                        gen_snippet = s["generated_text"].replace("\n", " ")
                        f.write(f"| {s['temperature']} | {gen_snippet} | {s['distinct_1']} | {s['distinct_2']} | {s['repetition_3']*100:.1f}% |\n")
                f.write("\n")

    # Format text summary
    txt_path = SAMPLES_DIR / "generated_samples_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("AUTOREGRESSIVE TEXT GENERATION - SAMPLE TRANSCRIPTS\n")
        f.write("=" * 80 + "\n\n")
        for s in detailed_samples:
            f.write(f"MODEL: {s['model']} | TEMP: {s['temperature']}\n")
            f.write(f"SEED: {s['seed']}\n")
            f.write(f"GENERATION: {s['generated_text']}\n")
            f.write(f"METRICS: Distinct-1={s['distinct_1']}, Distinct-2={s['distinct_2']}, Rep-3={s['repetition_3']*100:.1f}%\n")
            f.write("-" * 80 + "\n\n")

    print(f"\nGenerated samples saved to:\n  - {json_path}\n  - {md_path}\n  - {txt_path}")
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
