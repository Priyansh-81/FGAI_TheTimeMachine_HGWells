# Autoregressive Text Generation System
## Foundations of Generative AI (ICT4403) — Assignment 2

An empirical comparative study of **Simple RNN**, **Bidirectional RNN**, **LSTM**, and **GRU** neural architectures trained under controlled, comparable experimental conditions for word-level autoregressive text generation.

---

## Table of Contents
1. [Problem Statement & Objective](#1-problem-statement--objective)
2. [Dataset Description & Canonical Source Link](#2-dataset-description--canonical-source-link)
3. [Architectures & Model Implementations](#3-architectures--model-implementations)
4. [Environment Setup & Execution Steps](#4-environment-setup--execution-steps)
5. [Quantitative Results & Comparison](#5-quantitative-results--comparison)
6. [Qualitative Analysis & Sampling-Temperature Evaluation](#6-qualitative-analysis--sampling-temperature-evaluation)
7. [Conclusion & Best-Performing Model Justification](#7-conclusion--best-performing-model-justification)
8. [Repository Structure & Submission Artifacts](#8-repository-structure--submission-artifacts)

---

## 1. Problem Statement & Objective

Autoregressive language models predict the conditional probability distribution of the next word given a historical sequence of context words:

$$P(W) = \prod_{t=1}^T P(w_t \mid w_{t-L}, \dots, w_{t-1})$$

The objective of this assignment is to:
1. Construct a clean word-level sliding-window dataset from an unconstrained literary corpus (*The Time Machine* by H. G. Wells).
2. Implement four fundamental recurrent neural network architectures:
   - **Simple RNN** (Unidirectional vanilla recurrent network)
   - **Bidirectional RNN** (Dual-direction context encoder over the prompt window)
   - **Long Short-Term Memory (LSTM)** (Gated additive cell highway)
   - **Gated Recurrent Unit (GRU)** (Streamlined update/reset gated unit)
3. Train all four models under strictly identical, comparable conditions (embedding size, recurrent dimensionality, dropout, optimizer, learning rate, loss, batch size, and epoch count).
4. Systematically evaluate both **quantitative metrics** (Cross-Entropy Loss, Perplexity, Top-1 and Top-5 Prediction Accuracy, Training Time, and Inference Latency) and **qualitative characteristics** (Coherence, Lexical Diversity, Repetition Rate, and Contextual Consistency).
5. Empirically investigate the influence of **sampling temperature** ($T \in \{0.2, 0.7, 1.2\}$) on generated text and identify the best-performing model.

---

## 2. Dataset Description & Canonical Source Link

- **Corpus**: *The Time Machine* by H. G. Wells (1895)
- **Source Link**: [Project Gutenberg eBook #35](https://www.gutenberg.org/cache/epub/35/pg35.txt)
- **Genre & Style**: Victorian speculative science fiction; rich in temporal vocabulary, philosophical reflections, descriptive pacing, and dialogue.

### Text Characteristics & Preprocessing Pipeline
1. **Header & Footer Stripping**: Project Gutenberg ASCII license text, disclaimers, and metadata markers were automatically identified and removed, leaving the raw body narrative.
2. **Text Normalization**: All tokens converted to lowercase; punctuation cleaned while preserving English contractions (e.g., `"don't"`, `"it's"`).
3. **Corpus Statistics**:
   - Total raw word tokens: **32,896**
   - Clean narrative characters: **179,698**
   - Unique raw vocabulary words: **4,597**
4. **Vocabulary Construction**:
   - Vocabulary size $|V| = 2,500$ words (Top 2,498 frequent words + reserved `<pad>` index `0` + reserved `<unk>` index `1`).
   - Achieved **93.63%** total token coverage over the entire text.
5. **Sequence Formatting**:
   - Sliding context window of length $L = 20$ tokens predicting token $21$.
   - Yielded **32,876** autoregressive input-target sequence pairs.
   - Partitioned into **85% Training** (27,944 pairs) and **15% Validation** (4,932 pairs) using a deterministic random seed (`42`).

---

## 3. Architectures & Model Implementations

To ensure fair empirical evaluation, all four architectures use strictly controlled identical parameters:
- **Embedding Dimension**: 128
- **Recurrent Dimension**: 128 (BiRNN uses 64 forward + 64 backward = 128 total concatenated units)
- **Dropout Rate**: 0.2
- **Dense Output Head**: Dense(2500, activation='softmax')
- **Optimizer**: Adam ($\text{learning rate} = 0.002$)
- **Loss Function**: Sparse Categorical Cross-Entropy
- **Batch Size**: 64
- **Training Epochs**: 15

### Architecture Summary Table

| Model | Architecture Highlights | Trainable Parameters | Latent Representation |
|:---|:---|:---:|:---:|
| **Simple RNN** | Vanilla recurrent transitions; direct $\tanh$ recurrence | **675,396** | 128 |
| **Bidirectional RNN** | 64 forward units + 64 backward units concatenated over prompt window | **667,204** | 128 (64 + 64) |
| **LSTM** | Input, Forget, Output gates + additive linear cell state highway | **774,084** | 128 |
| **GRU** | Reset and Update gates; coupled hidden update | **741,572** | 128 |

---

## 4. Environment Setup & Execution Steps

### Prerequisites
- Python 3.10+ (macOS / Linux / Windows)
- Virtual environment recommended

### Installation
```bash
# Clone the repository
git clone https://github.com/[YourUsername]/FGAI-Assignment2-Text-Generation.git
cd FGAI-Assignment2-Text-Generation

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Reproducing the Experiments
```bash
# 1. Run complete end-to-end training, evaluation, and temperature generation pipeline
python run_experiments.py

# 2. Launch the interactive Jupyter notebook
jupyter notebook assignment2_text_generation.ipynb
```

---

## 5. Quantitative Results & Comparison

### Quantitative Metrics Table (15 Epochs on Unseen Validation Set)

| Model | Parameters | Train Loss | Val Loss | Best Val Loss | Train PPL | Val PPL | Best Val PPL | Train Acc (%) | Val Acc (%) | Val Top-5 Acc (%) | Train Time (s) | Inference (ms/token) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Simple RNN** | 675,396 | 2.7564 | 6.3778 | 5.3802 | 15.74 | 588.61 | 217.06 | 37.00% | 13.06% | 28.53% | **59.68s** | **11.77 ms** |
| **Bidirectional RNN** | 667,204 | **1.8923** | 6.9442 | 5.3265 | **6.63** | 1037.13 | 205.71 | **55.56%** | 9.75% | 23.74% | 65.96s | 22.52 ms |
| **LSTM** | 774,084 | 3.2846 | **6.0029** | 5.3389 | 26.70 | **404.58** | 208.28 | 28.80% | **15.59%** | **32.22%** | 121.78s | 15.87 ms |
| **GRU** | 741,572 | 2.1791 | 6.4087 | **5.2428** | 8.84 | 607.13 | **189.19** | 48.43% | 11.46% | 28.49% | 120.37s | 20.24 ms |

### Key Quantitative Findings:
1. **Next-Word Prediction Accuracy**:
   - **LSTM** achieved the highest generalization performance with **15.59% Top-1 Accuracy** and **32.22% Top-5 Accuracy**, outperforming Simple RNN (13.06% Top-1) and GRU (11.46% Top-1).
2. **Loss and Perplexity**:
   - **LSTM** attained the lowest final validation cross-entropy loss (**6.0029**) and lowest final validation perplexity (**404.58**).
   - **GRU** reached the lowest transient validation loss at early epochs (**5.2428**, Perplexity **189.19**).
3. **Training Speed & Efficiency**:
   - **Simple RNN** trained the fastest (**59.68s** total, ~4s/epoch) due to absence of gating operations.
   - **Bidirectional RNN** trained in **65.96s**, while **LSTM** and **GRU** required **~120s** due to their internal gate matrices.
4. **Overfitting Dynamics in Bidirectional RNN**:
   - The Bidirectional RNN achieved an impressive training accuracy of **55.56%** and training loss of **1.8923**, but overfit noticeably on validation data (Val Loss 6.9442). Because the backward pass aggregates future prompt tokens, the model relies heavily on bidirectional prompt structure which differs from unidirectional test-time generation dynamics.

---

## 6. Qualitative Analysis & Sampling-Temperature Evaluation

Text was autoregressively generated from four fixed seed prompts using temperature-scaled softmax sampling:

$$p'_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

### Linguistic Metrics across Temperatures

| Model | Temperature ($T$) | Distinct-1 (Diversity) | Distinct-2 (Bigram) | Repetition-3 (%) | Consecutive Repetition (%) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Simple RNN** | 0.2 | 0.7714 | 0.9633 | 0.0% | 1.47% |
| **Simple RNN** | 0.7 | 0.8286 | 0.9780 | 0.76% | 0.0% |
| **Simple RNN** | 1.2 | 0.8714 | 0.9927 | 0.0% | 0.0% |
| **Bidirectional RNN** | 0.2 | 0.8000 | 0.9853 | 0.0% | 0.0% |
| **Bidirectional RNN** | 0.7 | 0.8428 | 0.9706 | 0.76% | 0.0% |
| **Bidirectional RNN** | 1.2 | 0.8714 | 1.0000 | 0.0% | 0.0% |
| **LSTM** | 0.2 | 0.6500 | 0.8235 | 8.33% | 0.0% |
| **LSTM** | **0.7 (Optimal)** | **0.8143** | **0.9853** | **0.0%** | **0.73%** |
| **LSTM** | 1.2 | 0.8714 | 0.9927 | 0.0% | 0.73% |
| **GRU** | 0.2 | 0.7429 | 0.9044 | 6.82% | 1.47% |
| **GRU** | **0.7 (Optimal)** | **0.8214** | **0.9927** | **0.73%** | **0.0%** |
| **GRU** | 1.2 | 0.8928 | 0.9927 | 0.0% | 0.0% |

### Generated Text Sample Excerpts
**Seed Prompt**: *"the time traveller began to explain the strange"*

- **Low Temperature ($T = 0.2$ - Greedy / Repetitive)**:
  - *LSTM*: `"...thing filby easier at right angles to the time traveller and the lamp and the lamp was very hot thing i saw the time traveller proceeded..."`
  - *Observation*: High grammatical structure but gets caught in localized repetitive phrases (`"the lamp and the lamp was very hot"`).
- **Moderate Temperature ($T = 0.7$ - Balanced / Optimal)**:
  - *LSTM*: `"...thing looking round me for a minute he walked over the machine he must have been him is a fourth dimension of his cigar a trick waned of cheerfulness he he said the lamp and..."`
  - *Observation*: Rich, evocative Wellsian terminology (`"fourth dimension"`, `"cigar"`, `"machine"`). High diversity (Distinct-1: 0.81) and zero trigram loops.
- **High Temperature ($T = 1.2$ - Creative / Disordered)**:
  - *LSTM*: `"...flowers sitting in the summit of the certain travelling worn too too again touched me to break speed i had most of ruins i had seen forthwith to open centuries but these other not followed..."`
  - *Observation*: High vocabulary entropy (Distinct-1: 0.87+), but syntactical collapse into fragmented and ungrammatical token sequences.

---

## 7. Conclusion & Best-Performing Model Justification

### Justification of Best Model: **Long Short-Term Memory (LSTM)**
Among the four models evaluated, **LSTM is selected as the best-performing architecture** for autoregressive text generation on this dataset for the following reasons:

1. **Superior Unseen Generalization**: LSTM achieved the lowest final validation cross-entropy loss (**6.0029**) and lowest final validation perplexity (**404.58**), proving superior generalization to new contexts.
2. **Highest Next-Word Prediction Accuracy**: LSTM scored **15.59% Top-1 Accuracy** and **32.22% Top-5 Accuracy**, significantly higher than Simple RNN (13.06%) and BiRNN (9.75%).
3. **Effective Memory Gating**: The additive cell state highway ($C_t$) protects error gradients from vanishing over the 20-word context window, allowing the network to retain distant grammatical cues (subject-verb agreement and thematic nouns).
4. **Qualitative Output Excellence**: At sampling temperature $T = 0.7$, LSTM produced the most contextually consistent and linguistically compelling continuations adhering to the narrative style of *The Time Machine*.

---

## 8. Repository Structure & Submission Artifacts

```
FGAI/
├── data/
│   ├── time_machine_raw.txt              # Downloaded raw Project Gutenberg text
│   ├── time_machine_clean.txt            # Preprocessed narrative body
│   └── vocab.json                        # Vocabulary mapping and metadata
├── models/
│   ├── simple_rnn.keras                  # Saved Simple RNN model
│   ├── bidirectional_rnn.keras           # Saved Bidirectional RNN model
│   ├── lstm.keras                        # Saved LSTM model (Best Model)
│   └── gru.keras                         # Saved GRU model
├── results/
│   ├── quantitative_metrics.csv          # Summary table of all quantitative measures
│   ├── training_history.csv              # Per-epoch loss and accuracy records
│   ├── temperature_diversity_metrics.csv # Diversity & repetition metrics across T
│   ├── training_curves.png               # 4-panel training trajectory plot
│   ├── perplexity_comparison.png         # Validation perplexity comparison bar plot
│   ├── accuracy_comparison.png           # Top-1 and Top-5 accuracy bar plot
│   ├── training_time_comparison.png      # Wall-clock training time bar plot
│   └── temperature_diversity_analysis.png# Distinct-1 and repetition rate vs T
├── samples/
│   ├── generated_samples.json            # Machine-readable generated texts & metrics
│   ├── samples_by_temperature.md         # Formatted Markdown catalog of generations
│   └── generated_samples_summary.txt     # Human-readable generation transcripts
├── src/
│   ├── __init__.py
│   ├── config.py                         # Experimental hyperparameters & paths
│   ├── dataset.py                        # Cleaning, tokenization & sequence generator
│   ├── models.py                         # Simple RNN, BiRNN, LSTM, GRU model definitions
│   ├── evaluate.py                       # Quantitative metrics & visualization functions
│   └── generator.py                      # Autoregressive generation & diversity metrics
├── assignment2_text_generation.ipynb     # Complete, executed Jupyter Notebook
├── run_experiments.py                    # End-to-end experiment pipeline script
├── requirements.txt                      # Dependencies list
└── README.md                             # Comprehensive project documentation
```
