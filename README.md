# Movie Review Sentiment Analysis

Binary sentiment classification of movie reviews (**positive** / **negative**) using sequence
models of increasing expressive power. The project compares four architectures — a simple RNN,
a GRU, a token-wise MLP with global average pooling, and a restricted (local) self-attention
layer — on the same data and training setup, and analyzes *why* each one succeeds or fails.

## Highlights

- Each review is truncated to **100 words**; every word is embedded with **frozen 100-dim GloVe** vectors.
- All models output **2 logits** so per-word sub-prediction scores can be averaged.
- Detailed **error analysis** (TP / TN / FP / FN) with per-word score tables.
- A focused study of **context**: the MLP scores words in isolation, the GRU captures
  long-range dependencies, and local attention captures short-range (adjacent-word) reversals.

## Results

| Strategy                          | Test Accuracy | Notes |
| --------------------------------- | :-----------: | ----- |
| Simple RNN                        | ~0.50–0.60    | Fails — vanishing gradient over 100 steps |
| GRU                               | **~0.83**     | Gates enable long-range dependencies |
| Token-wise MLP + Global Avg Pool  | **0.828**     | Strong baseline; no context |
| Restricted Self-Attention         | **0.827**     | Same accuracy, but contextualizes per-word scores |

See [`REPORT.md`](REPORT.md) for the full write-up, training curves, and analysis.


## Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/movie-review-sentiment-analysis.git
cd movie-review-sentiment-analysis

# (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Main dependencies: `torch`, `numpy`, `matplotlib`. You will also need the **GloVe** embeddings
(100-dim) available locally for the preprocessing step.

## Usage

Train a model by selecting the architecture and hidden size:

```bash
# Recurrent models
python src/train.py --model rnn --hidden 64
python src/train.py --model gru --hidden 128

# Token-wise MLP + Global Average Pooling
python src/train.py --model mlp --hidden 128

# Restricted local self-attention
python src/train.py --model attention --hidden 128
```

Run the error analysis (per-word sub-prediction tables for the TP / TN / FP / FN examples):

```bash
python src/analysis.py --model mlp
python src/analysis.py --model attention
```


## The four strategies

1. **Simple RNN (Elman cell).** Concatenates the word vector with the previous hidden state,
   passes through a linear layer + `tanh`. The final hidden state is classified. Struggles with
   long sequences due to vanishing gradients.
2. **GRU.** Adds learnable **reset** and **update** gates, letting information and gradients
   flow across long sequences and solving the long-range dependency problem.
3. **Token-wise MLP + Global Average Pooling.** Scores each word independently with a small MLP,
   then averages the per-word logits. Strong but context-free.
4. **Restricted Self-Attention.** Single-head attention over a local window of 11 words
   (5 each side) with a learnable local positional encoding. Contextualizes each word using its
   neighbors, capturing short-range reversals such as negation.

## License

 MIT `LICENSE`.
