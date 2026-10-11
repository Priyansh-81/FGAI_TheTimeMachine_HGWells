"""
Model architectures for Autoregressive Text Generation:
1. Simple RNN
2. Bidirectional RNN
3. Long Short-Term Memory (LSTM)
4. Gated Recurrent Unit (GRU)

All models are designed under strictly comparable experimental conditions:
- Identical vocabulary size
- Identical embedding dimension (128)
- Identical recurrent representation dimension (128 total)
- Identical dropout regularization (0.2)
- Identical output classification head (Dense with Softmax)
- Identical optimization parameters (Adam, lr=0.002, sparse categorical crossentropy)
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from src.config import DROPOUT_RATE, EMBED_DIM, HIDDEN_DIM, LEARNING_RATE, SEQ_LENGTH, VOCAB_SIZE


def compile_model(model: tf.keras.Model, learning_rate: float = LEARNING_RATE) -> tf.keras.Model:
    """Compile model with Adam optimizer and Sparse Categorical Crossentropy loss."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_accuracy")
        ]
    )
    return model


def build_simple_rnn(
    vocab_size: int = VOCAB_SIZE,
    embed_dim: int = EMBED_DIM,
    hidden_dim: int = HIDDEN_DIM,
    seq_length: int = SEQ_LENGTH,
    dropout: float = DROPOUT_RATE,
    learning_rate: float = LEARNING_RATE
) -> tf.keras.Model:
    """
    Standard unidirectional Simple RNN architecture for autoregressive generation.
    Processes the sequence causally from left to right.
    """
    model = models.Sequential([
        layers.Input(shape=(seq_length,), dtype="int32", name="input_tokens"),
        layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=False, name="embedding"),
        layers.SimpleRNN(hidden_dim, dropout=dropout, return_sequences=False, name="simple_rnn"),
        layers.Dense(vocab_size, activation="softmax", name="output_logits")
    ], name="Simple_RNN")
    return compile_model(model, learning_rate)


def build_bidirectional_rnn(
    vocab_size: int = VOCAB_SIZE,
    embed_dim: int = EMBED_DIM,
    hidden_dim: int = HIDDEN_DIM,
    seq_length: int = SEQ_LENGTH,
    dropout: float = DROPOUT_RATE,
    learning_rate: float = LEARNING_RATE
) -> tf.keras.Model:
    """
    Bidirectional Simple RNN architecture.
    Processes the known causal prompt window in both directions (forward and backward)
    to form a richer contextual summary of the preceding history.
    Uses hidden_dim // 2 for each direction so that the concatenated representation
    matches the 128-dimensional latent space of the unidirectional models.
    """
    half_hidden = hidden_dim // 2
    model = models.Sequential([
        layers.Input(shape=(seq_length,), dtype="int32", name="input_tokens"),
        layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=False, name="embedding"),
        layers.Bidirectional(
            layers.SimpleRNN(half_hidden, dropout=dropout, return_sequences=False),
            name="bidirectional_rnn"
        ),
        layers.Dense(vocab_size, activation="softmax", name="output_logits")
    ], name="Bidirectional_RNN")
    return compile_model(model, learning_rate)


def build_lstm(
    vocab_size: int = VOCAB_SIZE,
    embed_dim: int = EMBED_DIM,
    hidden_dim: int = HIDDEN_DIM,
    seq_length: int = SEQ_LENGTH,
    dropout: float = DROPOUT_RATE,
    learning_rate: float = LEARNING_RATE
) -> tf.keras.Model:
    """
    Long Short-Term Memory (LSTM) architecture.
    Equipped with input, forget, and output gates and a cell state highway
    to mitigate vanishing/exploding gradients and capture long-range dependencies.
    """
    model = models.Sequential([
        layers.Input(shape=(seq_length,), dtype="int32", name="input_tokens"),
        layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=False, name="embedding"),
        layers.LSTM(hidden_dim, dropout=dropout, return_sequences=False, name="lstm"),
        layers.Dense(vocab_size, activation="softmax", name="output_logits")
    ], name="LSTM")
    return compile_model(model, learning_rate)


def build_gru(
    vocab_size: int = VOCAB_SIZE,
    embed_dim: int = EMBED_DIM,
    hidden_dim: int = HIDDEN_DIM,
    seq_length: int = SEQ_LENGTH,
    dropout: float = DROPOUT_RATE,
    learning_rate: float = LEARNING_RATE
) -> tf.keras.Model:
    """
    Gated Recurrent Unit (GRU) architecture.
    Streamlined gated mechanism with update and reset gates, offering comparable
    long-term memory retention to LSTM with fewer parameters and faster execution.
    """
    model = models.Sequential([
        layers.Input(shape=(seq_length,), dtype="int32", name="input_tokens"),
        layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=False, name="embedding"),
        layers.GRU(hidden_dim, dropout=dropout, return_sequences=False, name="gru"),
        layers.Dense(vocab_size, activation="softmax", name="output_logits")
    ], name="GRU")
    return compile_model(model, learning_rate)


def get_all_models(vocab_size: int = VOCAB_SIZE) -> dict:
    """Instantiate all 4 models for comparative evaluation."""
    return {
        "Simple_RNN": build_simple_rnn(vocab_size=vocab_size),
        "Bidirectional_RNN": build_bidirectional_rnn(vocab_size=vocab_size),
        "LSTM": build_lstm(vocab_size=vocab_size),
        "GRU": build_gru(vocab_size=vocab_size),
    }
