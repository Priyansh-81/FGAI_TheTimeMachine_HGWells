"""
Dataset loading, text preprocessing, vocabulary creation, and sequence construction.
"""

import json
import re
import urllib.request
from collections import Counter
from typing import Dict, List, Tuple
import numpy as np


from src.config import (
    CLEAN_DATA_PATH,
    DATASET_URL,
    PAD_ID,
    PAD_TOKEN,
    RAW_DATA_PATH,
    RANDOM_SEED,
    SEQ_LENGTH,
    UNK_ID,
    UNK_TOKEN,
    VAL_SPLIT,
    VOCAB_PATH,
    VOCAB_SIZE,
)


def download_dataset() -> str:
    """Download the raw text of The Time Machine from Project Gutenberg if not already present."""
    if not RAW_DATA_PATH.exists():
        print(f"Downloading raw dataset from {DATASET_URL}...")
        urllib.request.urlretrieve(DATASET_URL, RAW_DATA_PATH)
        print(f"Saved raw text to {RAW_DATA_PATH}")
    else:
        print(f"Raw dataset already exists at {RAW_DATA_PATH}")

    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        return f.read()


def clean_gutenberg_text(raw_text: str) -> str:
    """
    Remove Project Gutenberg metadata headers, licensing footers,
    table of contents, and extraneous non-narrative markers.
    """
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"

    start_idx = raw_text.find(start_marker)
    if start_idx != -1:
        start_idx = raw_text.find("\n", start_idx) + 1
    else:
        start_idx = 0

    end_idx = raw_text.find(end_marker)
    if end_idx == -1:
        end_idx = len(raw_text)

    body = raw_text[start_idx:end_idx].strip()

    # Normalize whitespace and save clean text
    with open(CLEAN_DATA_PATH, "w", encoding="utf-8") as f:
        f.write(body)
    print(f"Clean narrative text saved to {CLEAN_DATA_PATH} ({len(body)} chars)")
    return body


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into lowercased word tokens using regex.
    Preserves contractions (e.g. "don't", "it's").
    """
    # Match alphanumeric words and apostrophes
    tokens = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text.lower())
    return tokens


def build_vocabulary(tokens: List[str], max_vocab_size: int = VOCAB_SIZE) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build word-to-index and index-to-word mappings.
    Reserves index 0 for <pad> and index 1 for <unk>.
    """
    freq = Counter(tokens)
    # Reserve 2 slots for <pad> and <unk>
    num_words_to_keep = max_vocab_size - 2
    most_common = freq.most_common(num_words_to_keep)

    word2idx = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    idx2word = {PAD_ID: PAD_TOKEN, UNK_ID: UNK_TOKEN}

    for idx, (word, count) in enumerate(most_common, start=2):
        word2idx[word] = idx
        idx2word[idx] = word

    # Save vocabulary metadata
    vocab_meta = {
        "vocab_size": len(word2idx),
        "total_corpus_tokens": len(tokens),
        "unique_raw_words": len(freq),
        "word2idx": word2idx,
        "idx2word": {str(k): v for k, v in idx2word.items()}
    }
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab_meta, f, indent=2)

    coverage = sum(c for w, c in most_common) / len(tokens)
    print(f"Vocabulary built: {len(word2idx)} entries | Corpus tokens: {len(tokens)} | Coverage: {coverage*100:.2f}%")
    return word2idx, idx2word


def load_vocabulary() -> Tuple[Dict[str, int], Dict[int, str]]:
    """Load existing vocabulary from VOCAB_PATH."""
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    word2idx = meta["word2idx"]
    idx2word = {int(k): v for k, v in meta["idx2word"].items()}
    return word2idx, idx2word


def create_sequences(tokens: List[str], word2idx: Dict[str, int], seq_len: int = SEQ_LENGTH) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding-window input sequences and corresponding target tokens.
    For token sequence [t_0, t_1, ..., t_N-1]:
    Input: [t_i, ..., t_{i+seq_len-1}]
    Target: t_{i+seq_len}
    """
    token_ids = [word2idx.get(w, UNK_ID) for w in tokens]

    inputs = []
    targets = []
    for i in range(len(token_ids) - seq_len):
        inputs.append(token_ids[i : i + seq_len])
        targets.append(token_ids[i + seq_len])

    X = np.array(inputs, dtype=np.int32)
    y = np.array(targets, dtype=np.int32)
    print(f"Constructed {len(X)} autoregressive sequence pairs (Sequence length: {seq_len})")
    return X, y


def prepare_dataset(val_split: float = VAL_SPLIT, random_seed: int = RANDOM_SEED):
    """
    Complete data pipeline: download, clean, tokenize, build vocabulary,
    create input-target sequences, and split into train and validation sets.
    """
    raw_text = download_dataset()
    clean_text = clean_gutenberg_text(raw_text)
    tokens = tokenize_text(clean_text)
    word2idx, idx2word = build_vocabulary(tokens, VOCAB_SIZE)
    X, y = create_sequences(tokens, word2idx, SEQ_LENGTH)

    # Deterministic train/val split
    np.random.seed(random_seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    split_idx = int(len(X) * (1 - val_split))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]

    print(f"Data split: Train={len(X_train)} samples, Validation={len(X_val)} samples (val_split={val_split})")
    return (X_train, y_train), (X_val, y_val), word2idx, idx2word, tokens
