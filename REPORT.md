# Movie Review Sentiment Analysis — Project Report

Sequence models for binary sentiment classification of movie reviews, comparing four
architectures of increasing expressive power: a simple recurrent cell, a gated recurrent
cell, a token-wise MLP with global average pooling, and a restricted (local) self-attention
layer.

## Overview

The goal is to predict the sentiment of a movie review (**positive** / **negative**) from
its text. Each review is truncated to **100 words**, and every word is mapped to a
**100-dimensional vector** using **frozen GloVe embeddings**. The embeddings are kept frozen
to avoid overfitting on a relatively small training set.

Four strategies are compared:

1. **Simple RNN** (Elman cell)
2. **GRU** (gated recurrent unit)
3. **Token-wise MLP + Global Average Pooling**
4. **Restricted (local) Self-Attention**

All models output **two logits** (one per sentiment class). For the pooling and attention
strategies this is essential, since per-word two-dimensional *sub-prediction scores* are
averaged to form the final prediction.

---

## Task 1 — Recurrent Networks (RNN & GRU)

### 1.1 The Elman (RNN) cell

At each time step the cell receives the word vector `x_t` and the previous hidden state
`h_{t-1}`, concatenates them, passes the result through a single linear layer, and applies
`tanh` to produce `h_t`. After scanning all 100 words, the final hidden state is passed
through a classification layer that produces 2 logits (one per sentiment class).

```python
class ExRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExRNN, self).__init__()
        self.hidden_size = hidden_size
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.hidden2out = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state):
        combined = torch.cat((x, hidden_state), dim=1)
        hidden = torch.tanh(self.in2hidden(combined))
        output = self.hidden2out(hidden)
        return output, hidden

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size,
                           device=next(self.parameters()).device)
```

### 1.2 The GRU cell

The GRU adds two learnable gates: an **update gate**, which controls how much of the old
state is preserved versus the new candidate, and a **reset gate**, which controls how much of
the previous state to "forget" when computing the candidate. These gates let information and
gradients flow across long sequences, mitigating the **vanishing gradient** problem.

```python
class ExGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExGRU, self).__init__()
        self.hidden_size = hidden_size
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.reset_gate  = nn.Linear(input_size + hidden_size, hidden_size)
        self.candidate   = nn.Linear(input_size + hidden_size, hidden_size)
        self.hidden2out  = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state):
        combined = torch.cat((x, hidden_state), dim=1)
        update = torch.sigmoid(self.update_gate(combined))
        reset  = torch.sigmoid(self.reset_gate(combined))
        reset_hidden = reset * hidden_state
        candidate_combined = torch.cat((x, reset_hidden), dim=1)
        candidate_hidden = torch.tanh(self.candidate(candidate_combined))
        hidden = (1 - update) * hidden_state + update * candidate_hidden
        output = self.hidden2out(hidden)
        return output, hidden
```

Both cells emit **2 logits** (rather than a single scalar) so that the same building blocks
can later be reused by the pooling and attention strategies, which average two-dimensional
per-word sub-prediction scores.

### Training setup

Each architecture (RNN, GRU) was trained with two hidden-state sizes — **small (64)** and
**large (128)**. Every run used:

| Setting        | Value          |
| -------------- | -------------- |
| Epochs         | 10             |
| Optimizer      | Adam           |
| Learning rate  | 0.001          |
| Loss function  | Cross-Entropy  |

### 1.3 RNN results

| RNN accuracy (hidden = 64) | RNN loss (hidden = 64) |
| :---: | :---: |
| ![RNN accuracy, hidden = 64](assets/rnn_accuracy_h64.png) | ![RNN loss, hidden = 64](assets/rnn_loss_h64.png) |

| RNN accuracy (hidden = 128) | RNN loss (hidden = 128) |
| :---: | :---: |
| ![RNN accuracy, hidden = 128](assets/rnn_accuracy_h128.png) | ![RNN loss, hidden = 128](assets/rnn_loss_h128.png) |

### 1.4 GRU results

| GRU accuracy (hidden = 64) | GRU loss (hidden = 64) |
| :---: | :---: |
| ![GRU accuracy, hidden = 64](assets/gru_accuracy_h64.png) | ![GRU loss, hidden = 64](assets/gru_loss_h64.png) |

| GRU accuracy (hidden = 128) | GRU loss (hidden = 128) |
| :---: | :---: |
| ![GRU accuracy, hidden = 128](assets/gru_accuracy_h128.png) | ![GRU loss, hidden = 128](assets/gru_loss_h128.png) |

### 1.5 Results summary

| Model | Hidden | Train Acc | Test Acc | Train Loss | Test Loss |
| :---: | :----: | :-------: | :------: | :--------: | :-------: |
| GRU   | 64     | 0.927     | 0.828    | 0.193      | 0.413     |
| GRU   | 128    | 0.9746    | 0.8256   | 0.0770     | 0.6122    |
| RNN   | 64     | 0.600     | 0.595    | 0.667      | 0.670     |
| RNN   | 128    | 0.5033    | 0.5028   | 0.7110     | 0.7129    |

### 1.6 Analysis

**GRU vs. RNN.** The GRU reaches a test accuracy of roughly **0.83**, while the plain RNN is
stuck around **0.50–0.60** — close to random guessing. The reason is the **vanishing gradient**
problem: in the RNN, the hidden state is multiplied repeatedly across 100 steps, and the
gradient decays before it reaches the early words, so the network never learns long-range
dependencies. The GRU solves this with its reset/update gates, which allow information to be
preserved across the sequence and irrelevant information to be selectively ignored.

**Effect of hidden size (64 vs. 128).** For the GRU, both sizes reach a similar test accuracy
(~0.83), but with `hidden = 128` the **overfitting is stronger**: train loss drops to **0.08**
while test loss rises from **0.37** (epoch 5) to **0.61** (epoch 10). With `hidden = 64` the
gap is smaller and test loss stabilizes around **0.37–0.41**. In other words, more parameters
mean higher capacity but also a stronger tendency to memorize the training set. For the RNN,
both sizes fail — increasing capacity does not help when the gradient itself cannot flow
across the sequence, so the problem is **structural rather than a capacity issue**.

### 1.7 Example review demonstrating the RNN/GRU difference

We constructed a probe review that tests **sentiment reversal across the sequence** (a
long-range dependency) — a property where the GRU is expected to succeed and the RNN to fail.
The idea: open the review with a positive sentence, then flip the sentiment later so that the
final conclusion is negative. To classify it correctly, the model must remember the reversal
and weight the end of the sequence appropriately.

The most representative result (ground-truth label: **negative**):

```
Review:
At first I liked the movie but after a while it became slow confusing and disappointing

RNN: positive   (positive = 0.5309, negative = 0.4691)
GRU: negative   (positive = 0.0248, negative = 0.9752)
```

**Analysis:** the review opens positively ("At first I liked the movie") and flips at the end
("slow confusing and disappointing"), so the overall sentiment is negative. The RNN stays
undecided (positive = 0.5309) and even predicts positive incorrectly — consistent with the
fact that it barely learns (test accuracy ≈ 0.50). The GRU predicts negative with high
confidence (0.9752): its gates let it update the state so that the negative end of the
sequence becomes dominant. The example illustrates the GRU's advantage in handling sentiment
reversal and long-range dependencies.

### 1.8 RNN — combining the input with the hidden state and applying the activation

```python
combined = torch.cat((x, hidden_state), dim=1)
hidden = torch.tanh(self.in2hidden(combined))
```

### 1.9 GRU — computing the reset gate and update gate

```python
update = torch.sigmoid(self.update_gate(combined))
reset  = torch.sigmoid(self.reset_gate(combined))
```

---

## Task 2 — Token-wise MLP + Global Average Pooling

In this strategy each word passes independently through a small MLP that produces a
two-dimensional logit vector — a per-word **sub-prediction score**. All vectors are then
averaged to form the final prediction, which naturally handles variable-length input.

### 2.1 The FC architecture

We used a **three-layer MLP** with ReLU activations between layers:
`input → 128 → 128 → 2`. (We tried several hidden-layer sizes with almost no difference and
chose this one because it was marginally better than the alternatives.) The final layer
produces 2 logits per word. With `hidden = 128` a low test loss of **0.38** was achieved.

```python
class ExMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExMLP, self).__init__()
        self.ReLU = torch.nn.ReLU()
        self.layer1 = MatMul(input_size, hidden_size)
        self.layer2 = MatMul(hidden_size, hidden_size)
        self.layer3 = MatMul(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.ReLU(x)
        x = self.layer2(x)
        x = self.ReLU(x)
        x = self.layer3(x)
        return x
```

### 2.2 Where to apply softmax, and the Global Average Pooling line

**Answer: after averaging** — we average the logits of all words, and apply softmax only to
the averaged vector.

**Rationale:** averaging the logits preserves the signal strength of informative words. If we
applied softmax to each word separately *before* averaging, we would squash every score into
the `[0,1]` range and lose the distinction between a word with an extreme logit (e.g.
*"wonderful"* with +39) and a weak word — both would contribute almost equally. Averaging the
logits gives strong words greater weight, and is also consistent with the fact that
`CrossEntropyLoss` expects logits.

The line performing Global Average Pooling:

```python
mask = (reviews.abs().sum(dim=-1) > 0).float()
expanded_mask = mask.unsqueeze(-1)
masked_sub_scores = sub_score * expanded_mask
output = masked_sub_scores.sum(dim=1) / mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
```

### 2.3 Training results (hidden = 128)

| MLP accuracy (hidden = 128) | MLP loss (hidden = 128) |
| :---: | :---: |
| ![MLP accuracy, hidden = 128](assets/mlp_accuracy_h128.png) | ![MLP loss, hidden = 128](assets/mlp_loss_h128.png) |

Full evaluation after training:

```
Train Loss: 0.3108, Train Acc: 0.8649
Test Loss:  0.3820, Test Acc:  0.8277
```

The MLP reaches a test accuracy of **0.828** — comparable to the GRU — with only mild
overfitting (the gap between train and test loss is small). This is achieved even though the
model scores each word independently and without context, which shows that for sentiment
analysis a substantial part of the signal is carried at the single-word level.

### 2.4 Error analysis — sub-prediction score tables

For each review we print a table of (word, positive score, negative score). The final
prediction is the average of the scores over all words.

#### Review 1 — TP (true positive)

> *"This movie was wonderful and exciting with great acting and a beautiful story"*
> Ground truth: **positive** · Prediction: **positive** (positive = 1.0000, negative = 0.0000)

| Word      | pos_score | neg_score |
| --------- | --------: | --------: |
| this      | 0.2064    | -0.1290   |
| movie     | -0.4473   | 0.4382    |
| was       | -1.3385   | 1.4441    |
| wonderful | 39.4231   | -42.2850  |
| and       | 3.6693    | -3.7900   |
| exciting  | 10.0295   | -11.1929  |
| with      | 0.8733    | -0.7203   |
| great     | 22.5985   | -23.7080  |
| acting    | -3.5750   | 3.6496    |
| and       | 3.6693    | -3.7900   |
| beautiful | 6.7605    | -7.0589   |
| story     | 1.1143    | -1.1441   |

#### Review 2 — TN (true negative)

> *"This movie was boring slow confusing and disappointing with terrible acting"*
> Ground truth: **negative** · Prediction: **negative** (positive = 0.0000, negative = 1.0000)

| Word         | pos_score | neg_score |
| ------------ | --------: | --------: |
| this         | 0.2064    | -0.1290   |
| movie        | -0.4473   | 0.4382    |
| was          | -1.3385   | 1.4441    |
| boring       | -57.3529  | 56.0766   |
| slow         | -20.8055  | 19.9113   |
| confusing    | -15.9973  | 16.9740   |
| and          | 3.6693    | -3.7900   |
| disappointing| -66.9471  | 64.1545   |
| with         | 0.8733    | -0.7203   |
| terrible     | -55.7665  | 53.6169   |
| acting       | -3.5750   | 3.6496    |

#### Review 3 — FP (false positive)

> *"My friends said the movie was wonderful but my experience was the complete opposite"*
> Ground truth: **negative** · Prediction: **positive** (positive = 0.9998, negative = 0.0002)

| Word       | pos_score | neg_score |
| ---------- | --------: | --------: |
| my         | 4.9463    | -5.8035   |
| friends    | 8.2292    | -9.2024   |
| said       | 0.3514    | -0.3365   |
| the        | 0.1188    | -0.1086   |
| movie      | -0.2844   | 0.0286    |
| was        | -1.1113   | 1.0442    |
| wonderful  | 36.0679   | -40.5629  |
| but        | 0.6312    | -0.5542   |
| my         | 4.9463    | -5.8035   |
| experience | 13.8328   | -15.5956  |
| was        | -1.1113   | 1.0442    |
| the        | 0.1188    | -0.1086   |
| complete   | -9.8351   | 9.1485    |
| opposite   | -1.1884   | 0.7348    |

#### Review 4 — FN (false negative)

> *"The beginning was slow and boring but later the story became moving beautiful and excellent"*
> Ground truth: **positive** · Prediction: **negative** (positive = 0.2869, negative = 0.7131)

| Word      | pos_score | neg_score |
| --------- | --------: | --------: |
| the       | -0.0832   | 0.1227    |
| beginning | 2.4903    | -2.0274   |
| was       | -1.3385   | 1.4441    |
| slow      | -20.8055  | 19.9113   |
| and       | 3.6693    | -3.7900   |
| boring    | -57.3529  | 56.0766   |
| but       | 1.0691    | -0.5551   |
| later     | 2.9926    | -2.4629   |
| the       | -0.0832   | 0.1227    |
| story     | 1.1143    | -1.1441   |
| became    | 1.0359    | -0.9281   |
| moving    | 0.3254    | -0.4023   |
| beautiful | 6.7605    | -7.0589   |
| and       | 3.6693    | -3.7900   |
| excellent | 48.0875   | -50.3075  |

**Explanation of the results.** The MLP scores each word independently, so words with a strong
emotional charge dominate the average: *"wonderful"* (+39), *"excellent"* (+48) on one side,
and *"terrible"* (−56), *"disappointing"* (−67), *"boring"* (−57) on the other. Function words
(*"and"*, *"was"*, *"the"*) receive near-zero scores and barely contribute.

In TP and TN the dominant words match the label, so the prediction is correct and
high-confidence.

In **FP** (review 3) the review is actually negative ("my experience was the complete
opposite" — the opposite of "wonderful"), yet the model predicts positive with high
confidence. The cause is the same: *"wonderful"* (+36), *"experience"* (+13.8) and *"friends"*
(+8.2) carry high positive scores, while the negating words *"complete"* (−9.8) and
*"opposite"* (−1.2) are too weak to balance them. The MLP does not understand that the phrase
*"the complete opposite"* inverts the meaning of *"wonderful"* — again, due to the absence of
context.

In **FN** (review 4) the same limitation appears from the opposite direction: the review is
actually positive ("became moving beautiful and excellent"), but the negative words at the
opening — *"boring"* (−57) and *"slow"* (−21) — are large in absolute value and drag the
average negative, despite *"excellent"* (+48). The MLP does not see that *"but later"* reverses
the meaning of the opening.

Both errors (FP and FN) stem from the MLP scoring words independently and without order —
exactly the problem that the **Self-Attention layer (Task 3)** is designed to solve.

---

## Task 3 — Restricted Self-Attention

### 3.1 Implementation of the Self-Attention layer

We implemented the layer following the `ExLRestSelfAtten` skeleton: each word first passes
through an FC layer (`layer1`) to `hidden`, then a **local neighborhood of 11 words** (5 on
each side) is built using `torch.roll` and zero padding. A **learnable local positional
encoding** is added, and single-head attention is computed (Q, K, V). The current word is the
**Query**, and its neighbors are the **Keys/Values**; the output is a weighted average of the
Values within the window.

```python
class ExLRestSelfAtten(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.atten_size = atten_size
        self.window_size = 2 * atten_size + 1
        self.sqrt_hidden_size = np.sqrt(float(hidden_size))
        self.ReLU = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=2)
        self.layer1 = MatMul(input_size, hidden_size)
        # learnable local positional encoding for offsets [-atten_size..+atten_size]
        self.pos_encoding = nn.Parameter(
            0.01 * torch.randn(1, 1, self.window_size, hidden_size))
        self.W_q = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_k = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_v = MatMul(hidden_size, hidden_size, use_bias=False)
        self.layer2 = MatMul(hidden_size, hidden_size)
        self.layer3 = MatMul(hidden_size, output_size)

    def forward(self, x):
        x = self.ReLU(self.layer1(x))                       # [B, L, H]
        padded = pad(x, (0, 0, self.atten_size, self.atten_size, 0, 0))
        x_nei = [torch.roll(padded, k, dims=1)
                 for k in range(-self.atten_size, self.atten_size + 1)]
        x_nei = torch.stack(x_nei, dim=2)
        x_nei = x_nei[:, self.atten_size:-self.atten_size, :, :]
        x_nei = x_nei + self.pos_encoding                   # local positional encoding
        query = self.W_q(x).unsqueeze(2)                    # [B, L, 1, H]
        keys  = self.W_k(x_nei)                             # [B, L, win, H]
        vals  = self.W_v(x_nei)
        atten_scores = torch.sum(query * keys, dim=-1) / self.sqrt_hidden_size
        atten_weights = self.softmax(atten_scores)
        x = torch.sum(atten_weights.unsqueeze(-1) * vals, dim=2)  # [B, L, H]
        x = self.ReLU(self.layer2(x))
        x = self.layer3(x)                                  # [B, L, 2] sub-scores
        return x, atten_weights
```

The lines that compute the attention scores (Query · Keys):

```python
query = self.W_q(x).unsqueeze(2)                       # [B, L, 1, H]
keys  = self.W_k(x_nei)                                # [B, L, win, H]
atten_scores = torch.sum(query * keys, dim=-1) / self.sqrt_hidden_size
atten_weights = self.softmax(atten_scores)
```

---

## Task 4 — Self-Attention Training Results (hidden = 128)

| Attention accuracy (hidden = 128) | Attention loss (hidden = 128) |
| :---: | :---: |
| ![Attention accuracy, hidden = 128](assets/attention_accuracy_h128.png) | ![Attention loss, hidden = 128](assets/attention_loss_h128.png) |

Full evaluation after training:

```
Train Loss: 0.2792, Train Acc: 0.8798
Test Loss:  0.3946, Test Acc:  0.8273
```

The test accuracy (**0.827**) is very close to the MLP without attention (**0.828**). The layer
does not improve overall accuracy, because most reviews are dominated anyway by strongly
emotional words — but, as shown below, it substantially changes the per-word scores
(**contextualization**).

### 4.1 Error analysis and comparison to the MLP

We ran the same four reviews (TP, TN, FP, FN) through the attention network and printed the
per-word sub-prediction scores.

#### Review 1 — TP (true positive)

> *"This movie was wonderful and exciting with great acting and a beautiful story"*
> Ground truth: **positive** · Prediction: **positive** (positive = 1.0000, negative = 0.0000)

| Word      | pos_score | neg_score |
| --------- | --------: | --------: |
| this      | 21.6949   | -14.3195  |
| movie     | 28.3746   | -18.6072  |
| was       | 12.4900   | -8.2310   |
| wonderful | 20.6487   | -13.7151  |
| and       | -0.0515   | 0.0499    |
| exciting  | 19.0439   | -12.6300  |
| with      | -0.5480   | 0.5732    |
| great     | 10.5889   | -7.1491   |
| acting    | 17.1194   | -11.3531  |
| and       | 1.5058    | -0.9541   |
| beautiful | 0.5498    | 0.0716    |
| story     | 6.1979    | -3.9304   |

#### Review 2 — TN (true negative)

> *"This movie was boring slow confusing and disappointing with terrible acting"*
> Ground truth: **negative** · Prediction: **negative** (positive = 0.0000, negative = 1.0000)

| Word         | pos_score | neg_score |
| ------------ | --------: | --------: |
| this         | -34.0396  | 21.0476   |
| movie        | -46.3682  | 29.0836   |
| was          | -38.3297  | 23.7634   |
| boring       | -38.0157  | 23.0479   |
| slow         | -35.8782  | 21.8283   |
| confusing    | -38.5979  | 23.2549   |
| and          | -0.1670   | 0.1725    |
| disappointing| -41.9704  | 26.4766   |
| with         | 0.2294    | -0.5888   |
| terrible     | -36.3424  | 22.1330   |
| acting       | -38.3240  | 24.2916   |

#### Review 3 — FP (false positive)

> *"My friends said the movie was wonderful but my experience was the complete opposite"*
> Ground truth: **negative** · Prediction: **positive** (positive = 0.9935, negative = 0.0065)

| Word       | pos_score | neg_score |
| ---------- | --------: | --------: |
| my         | -3.4787   | 2.0874    |
| friends    | -7.1874   | 4.3130    |
| said       | 0.1061    | -0.2603   |
| the        | -0.3593   | 0.3704    |
| movie      | 28.0324   | -18.3857  |
| was        | 0.8436    | -0.5129   |
| wonderful  | 20.0591   | -13.3271  |
| but        | 0.6206    | -0.3829   |
| my         | 5.8800    | -4.0502   |
| experience | 0.4859    | -0.2154   |
| was        | 2.1756    | -1.0920   |
| the        | -0.3827   | 0.3223    |
| complete   | -2.4484   | 1.8050    |
| opposite   | -1.8098   | 1.4221    |

#### Review 4 — FN (false negative)

> *"The beginning was slow and boring but later the story became moving beautiful and excellent"*
> Ground truth: **positive** · Prediction: **negative** (positive = 0.0086, negative = 0.9914)

| Word      | pos_score | neg_score |
| --------- | --------: | --------: |
| the       | -3.4044   | 1.9916    |
| beginning | -16.5095  | 10.1426   |
| was       | -26.0837  | 15.9811   |
| slow      | -31.2057  | 19.3650   |
| and       | 0.3536    | -0.6872   |
| boring    | -33.7028  | 20.3774   |
| but       | 1.1625    | -1.5828   |
| later     | 0.3734    | -0.6061   |
| the       | 0.8325    | -0.8646   |
| story     | 3.3737    | -2.4400   |
| became    | 0.7954    | -0.9163   |
| moving    | 1.7626    | -1.2442   |
| beautiful | 31.1172   | -19.7178  |
| and       | 0.9385    | -0.6992   |
| excellent | 23.3677   | -14.7874  |

#### Explanation and comparison to the MLP

The most striking change relative to the MLP is that **neutral words** (*"was"*, *"movie"*,
*"this"*) now receive a strong score according to their **local context**: in TP they are
strongly positive (*"this"* +22, *"movie"* +28), and in TN strongly negative (*"movie"* −46,
*"this"* −34). In the MLP these same words received near-zero scores. This is the
**contextualization** that attention provides — each word absorbs the emotional charge of the
11-word window around it.

However, **FP and FN still fail**. The reason: the attention here is **local** (a window of 5
words on each side). In **FP** (*"...wonderful but my experience was the complete opposite..."*)
the word *"wonderful"* and the negating *"opposite"* are more than 5 words apart, so the window
does not bridge them, and *"wonderful"* (+20) remains dominant. In **FN** (*"...slow and boring
but later ... beautiful and excellent"*) the negative words at the opening and the positive
words at the end are far apart, and the *"but later"* reversal is not local. Long-range
sentiment reversals therefore remain out of reach for the local attention layer.

### 4.2 A context-dependent review — demonstrating the advantage of attention

To demonstrate a case where attention succeeds while the Task 2 MLP fails, we built a short
review in which a word changes its meaning because of its immediate neighbor:
**"The movie was not good"**. Here the word *"not"* negates *"good"*, and the reversal is
**local** (adjacent words), so it falls inside the attention window.

```
MLP network:        prediction positive (FP, WRONG)  - positive = 0.6260, negative = 0.3740
Attention network:  prediction negative (TN, CORRECT) - positive = 0.3320, negative = 0.6680
```

Per-word sub-prediction comparison (MLP vs. attention):

| Word  | MLP_pos | MLP_neg | ATT_pos | ATT_neg |
| ----- | ------: | ------: | ------: | ------: |
| the   | 0.1079  | -0.1067 | -0.0066 | -0.0681 |
| movie | -0.1127 | -0.0720 | -0.6389 | -0.2229 |
| was   | -1.1130 | 1.0179  | -1.0741 | 0.3613  |
| not   | -3.3802 | 2.8548  | -0.0462 | -0.0353 |
| good  | 4.8942  | -5.8736 | -1.2577 | 0.4373  |

The explanation lies in the score of the word *"good"*: in the MLP it receives a high positive
score (+4.89) regardless of context, so it overrides *"not"* (−3.38) and the prediction is
positive — wrong. In the attention network, by contrast, *"good"* receives a **negative** score
(−1.26 pos vs. +0.44 neg): its immediate neighbor *"not"* (one word away, inside the window)
flips its meaning. Note also that *"not"* itself nearly zeroes out under attention (−0.05),
because the negation effect has been "transferred" onto the word it negates. This is exactly
the capability the context-free MLP lacks, and here local attention succeeds where the MLP
fails.

---

## Summary

| Strategy                         | Test Acc | Key property |
| -------------------------------- | :------: | ------------ |
| Simple RNN                       | ~0.50–0.60 | Fails — vanishing gradient over 100 steps |
| GRU                              | ~0.83    | Gates enable long-range dependencies |
| Token-wise MLP + Global Avg Pool | 0.828    | Strong baseline; no context |
| Restricted Self-Attention        | 0.827    | Same accuracy, but contextualizes per-word scores |

The headline finding is that for this dataset a large fraction of the sentiment signal lives
at the single-word level, which is why the context-free MLP already matches the GRU. The value
of the gated and attention-based models shows up in the harder, context-dependent cases:
gates (GRU) handle long-range reversals, while local attention handles short-range
(adjacent-word) reversals such as negation.
