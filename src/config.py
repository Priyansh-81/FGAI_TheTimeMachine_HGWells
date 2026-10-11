"""
Configuration parameters for Autoregressive Text Generation System
Assignment 2 - Foundations of Generative AI (ICT4403)
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
SAMPLES_DIR = BASE_DIR / "samples"

# Ensure directories exist
for p in [DATA_DIR, MODELS_DIR, RESULTS_DIR, SAMPLES_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Dataset details
DATASET_NAME = "The Time Machine by H. G. Wells"
DATASET_URL = "https://www.gutenberg.org/cache/epub/35/pg35.txt"
RAW_DATA_PATH = DATA_DIR / "time_machine_raw.txt"
CLEAN_DATA_PATH = DATA_DIR / "time_machine_clean.txt"
VOCAB_PATH = DATA_DIR / "vocab.json"

# Special Tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1

# Hyperparameters (identical across all models for fair empirical comparison)
VOCAB_SIZE = 2500         # Top 2498 most frequent words + <pad> + <unk>
SEQ_LENGTH = 20           # Context window of preceding words
EMBED_DIM = 128           # Dense word embedding dimensionality
HIDDEN_DIM = 128          # Recurrent hidden state size
DROPOUT_RATE = 0.2        # Recurrent / post-layer dropout
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.002
VAL_SPLIT = 0.15          # 85% train, 15% validation
RANDOM_SEED = 42

# Generation parameters
TEMPERATURES = [0.2, 0.7, 1.2]  # Low (deterministic/repetitive), Medium (balanced), High (diverse/creative)
GENERATION_LENGTH = 35          # Number of tokens to autoregressively predict

SEED_SEQUENCES = [
    "the time traveller began to explain the strange",
    "it is simply that our consciousness moves along it",
    "i looked at the small machine and wondered how",
    "the world was completely changed in the distant future"
]
