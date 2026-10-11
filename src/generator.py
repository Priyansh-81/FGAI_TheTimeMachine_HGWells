"""
Autoregressive Text Generation and Qualitative Evaluation Module.
Supports temperature-scaled stochastic sampling and quantitative linguistic metrics
(Distinct-1, Distinct-2, Repetition Rate, and Self-Perplexity).
"""

import re
from typing import Dict, List, Tuple
import numpy as np
import tensorflow as tf

from src.config import GENERATION_LENGTH, PAD_ID, SEQ_LENGTH, TEMPERATURES, UNK_ID, UNK_TOKEN


def sample_with_temperature(probabilities: np.ndarray, temperature: float = 1.0) -> int:
    """
    Sample next token index using temperature scaling:
    p'_i = exp(log(p_i) / T) / sum_j exp(log(p_j) / T)
    """
    if temperature <= 1e-4:
        # Pure greedy argmax
        return int(np.argmax(probabilities))

    # Convert probability vector back to numerical logits
    probs = np.clip(probabilities, 1e-12, 1.0)
    logits = np.log(probs) / temperature
    exp_logits = np.exp(logits - np.max(logits))
    scaled_probs = exp_logits / np.sum(exp_logits)

    # Re-normalize to ensure sum is exactly 1.0
    scaled_probs = scaled_probs / np.sum(scaled_probs)
    return int(np.random.choice(len(scaled_probs), p=scaled_probs))


def generate_text(
    model: tf.keras.Model,
    seed_text: str,
    word2idx: Dict[str, int],
    idx2word: Dict[int, str],
    num_words_to_generate: int = GENERATION_LENGTH,
    temperature: float = 0.7,
    seq_length: int = SEQ_LENGTH,
    prevent_unk: bool = True
) -> Tuple[str, List[str], List[float]]:
    """
    Autoregressively generate next words given a seed prompt.
    Returns:
        full_text: Seed prompt + generated tokens joined by space.
        generated_tokens: List of newly generated tokens.
        token_probs: Probability of each chosen token at generation step.
    """
    # Tokenize seed prompt
    seed_tokens = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", seed_text.lower())
    seed_ids = [word2idx.get(w, UNK_ID) for w in seed_tokens]

    # Pad or truncate seed IDs to exact sequence length
    if len(seed_ids) < seq_length:
        curr_sequence = [PAD_ID] * (seq_length - len(seed_ids)) + seed_ids
    else:
        curr_sequence = seed_ids[-seq_length:]

    generated_tokens = []
    token_probs = []

    for _ in range(num_words_to_generate):
        input_array = np.array([curr_sequence[-seq_length:]], dtype=np.int32)
        preds = model(input_array, training=False).numpy()[0]

        if prevent_unk:
            # Mask out <pad> and <unk> to prevent degenerate outputs
            preds[PAD_ID] = 0.0
            preds[UNK_ID] = 0.0
            total_p = np.sum(preds)
            if total_p > 0:
                preds = preds / total_p

        next_idx = sample_with_temperature(preds, temperature=temperature)
        chosen_prob = float(preds[next_idx])
        token_probs.append(chosen_prob)

        next_word = idx2word.get(next_idx, UNK_TOKEN)
        generated_tokens.append(next_word)
        curr_sequence.append(next_idx)

    full_text = " ".join(seed_tokens + generated_tokens)
    return full_text, generated_tokens, token_probs


def compute_generation_metrics(tokens: List[str]) -> Dict[str, float]:
    """
    Compute linguistic diversity and repetition metrics for generated text:
    - Distinct-1 (unique unigrams / total unigrams)
    - Distinct-2 (unique bigrams / total bigrams)
    - Repetition-3 (fraction of repeated 3-grams)
    - Consecutive Repetition (fraction of tokens identical to predecessor)
    """
    if not tokens:
        return {"distinct_1": 0.0, "distinct_2": 0.0, "repetition_3": 0.0, "consecutive_rep": 0.0}

    total_unigrams = len(tokens)
    distinct_1 = len(set(tokens)) / total_unigrams

    # Bigrams
    if total_unigrams > 1:
        bigrams = [(tokens[i], tokens[i + 1]) for i in range(total_unigrams - 1)]
        distinct_2 = len(set(bigrams)) / len(bigrams)
    else:
        distinct_2 = 1.0

    # Trigram repetition
    if total_unigrams > 2:
        trigrams = [(tokens[i], tokens[i + 1], tokens[i + 2]) for i in range(total_unigrams - 2)]
        rep_3 = 1.0 - (len(set(trigrams)) / len(trigrams))
    else:
        rep_3 = 0.0

    # Consecutive repetition
    consec_count = sum(1 for i in range(len(tokens) - 1) if tokens[i] == tokens[i + 1])
    consec_rep = consec_count / max(1, len(tokens) - 1)

    return {
        "distinct_1": round(distinct_1, 4),
        "distinct_2": round(distinct_2, 4),
        "repetition_3": round(rep_3, 4),
        "consecutive_rep": round(consec_rep, 4),
    }


def generate_and_evaluate_samples(
    models_dict: Dict[str, tf.keras.Model],
    seed_sequences: List[str],
    word2idx: Dict[str, int],
    idx2word: Dict[int, str],
    temperatures: List[float] = TEMPERATURES,
    num_words: int = GENERATION_LENGTH
) -> Tuple[List[dict], List[dict]]:
    """
    Systematically generate samples for all models across seeds and temperatures.
    Computes qualitative and quantitative generation metrics.
    """
    detailed_samples = []
    aggregated_metrics = []

    for model_name, model in models_dict.items():
        for temp in temperatures:
            temp_distinct_1 = []
            temp_distinct_2 = []
            temp_rep_3 = []
            temp_consec = []

            for seed in seed_sequences:
                full_text, gen_tokens, probs = generate_text(
                    model=model,
                    seed_text=seed,
                    word2idx=word2idx,
                    idx2word=idx2word,
                    num_words_to_generate=num_words,
                    temperature=temp
                )
                metrics = compute_generation_metrics(gen_tokens)
                temp_distinct_1.append(metrics["distinct_1"])
                temp_distinct_2.append(metrics["distinct_2"])
                temp_rep_3.append(metrics["repetition_3"])
                temp_consec.append(metrics["consecutive_rep"])

                detailed_samples.append({
                    "model": model_name,
                    "temperature": temp,
                    "seed": seed,
                    "generated_text": " ".join(gen_tokens),
                    "full_text": full_text,
                    **metrics
                })

            aggregated_metrics.append({
                "model": model_name,
                "temperature": temp,
                "avg_distinct_1": round(float(np.mean(temp_distinct_1)), 4),
                "avg_distinct_2": round(float(np.mean(temp_distinct_2)), 4),
                "avg_repetition_3": round(float(np.mean(temp_rep_3)), 4),
                "avg_consecutive_rep": round(float(np.mean(temp_consec)), 4),
            })

    return detailed_samples, aggregated_metrics
