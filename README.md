# Sentiment Analysis: RNN, GRU, MLP, and Restricted Self-Attention

This repository contains the implementation and analysis of various neural network architectures for a binary sentiment analysis task (positive/negative) based on movie reviews. The project was completed as part of the "Introduction to Deep Learning" course.

## Project Overview
* **Input**: Reviews truncated to 100 words, where each word is mapped to a 100-dimensional vector using frozen GloVe embeddings.
* **Goal**: Predicting viewer sentiment (positive/negative).
* **Architectures Evaluated**:
    * **RNN (Elman Cell)**
    * **GRU (Gated Recurrent Unit)**
    * **MLP (Token-wise MLP)** with Global Average Pooling
    * **Restricted Self-Attention**

## Key Findings
* **RNN vs. GRU**: The vanilla RNN suffers from the vanishing gradient problem and fails to learn long-term dependencies. The GRU, utilizing learnable update/reset gates, effectively preserves information across the sequence and achieves significantly higher performance.
* **MLP Limitations**: While the MLP achieves competitive accuracy by focusing on highly emotional individual words, it lacks context awareness and struggles with negation or sentiment reversals (e.g., "not good").
* **Attention Benefits**: The Restricted Self-Attention mechanism introduces "contextualization"—neutral words (e.g., "movie," "was") absorb the emotional charge of their local neighborhood, allowing the model to resolve cases of local sentiment reversal that the MLP cannot handle.

## Repository Structure
* `ExRNN`: Implementation of the Elman RNN cell.
* `ExGRU`: Implementation of the GRU cell.
* `ExMLP`: Token-wise multi-layer perceptron architecture.
* `ExLRestSelfAtten`: Restricted self-attention layer with learnable local positional encoding.

## Documentation
For a detailed analysis of the results, performance graphs, error analysis, and theoretical questions regarding CNNs and attention mechanisms, please refer to the attached report.

## Contributors
* **Shimon Ifrach**
* **Yaakov Afgin**
