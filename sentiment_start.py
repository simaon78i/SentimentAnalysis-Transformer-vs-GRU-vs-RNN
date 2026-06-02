########################################################################
########################################################################
##                                                                    ##
##                      ORIGINAL _ DO NOT PUBLISH                     ##
##                                                                    ##
########################################################################
########################################################################

import torch as tr
import torch
from torch.nn.functional import pad
import torch.nn as nn
import numpy as np
import loader as ld
import matplotlib.pyplot as plt

batch_size = 32
output_size = 2
hidden_size = 64  # to experiment with

run_recurrent = False  # else run Token-wise MLP
use_RNN = True  # otherwise GRU
atten_size = 5  # atten > 0 means using restricted self atten

reload_model = False
num_epochs = 10
learning_rate = 0.001
test_interval = 50

# Loading sataset, use toy = True for obtaining a smaller dataset

train_dataset, test_dataset, num_words, input_size = ld.get_data_set(
    batch_size)


# Special matrix multipication layer (like torch.Linear but can operate on arbitrary sized
# tensors and considers its last two indices as the matrix.)

class MatMul(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias=True):
        super(MatMul, self).__init__()
        self.matrix = torch.nn.Parameter(torch.nn.init.xavier_normal_(
            torch.empty(in_channels, out_channels)), requires_grad=True)
        if use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(1, 1, out_channels),
                                           requires_grad=True)

        self.use_bias = use_bias

    def forward(self, x):
        x = torch.matmul(x, self.matrix)
        if self.use_bias:
            x = x + self.bias
        return x


# Implements RNN Unit

class ExRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExRNN, self).__init__()

        self.hidden_size = hidden_size

        # Elman RNN cell:
        # input:  [x_t, h_{t-1}]
        # output: h_t
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)

        # Classification layer:
        # input:  final hidden state h_t
        # output: 2 logits, one for each sentiment class
        self.hidden2out = nn.Linear(hidden_size, output_size)

    def name(self):
        return "RNN"

    def forward(self, x, hidden_state):
        # x shape:            [batch_size, input_size]
        # hidden_state shape: [batch_size, hidden_size]

        combined = torch.cat((x, hidden_state), dim=1)

        # hidden shape: [batch_size, hidden_size]
        hidden = torch.tanh(self.in2hidden(combined))

        # output shape: [batch_size, output_size] = [batch_size, 2]
        output = self.hidden2out(hidden)

        return output, hidden

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size,
                           device=next(self.parameters()).device)


# Implements GRU Unit

class ExGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExGRU, self).__init__()

        self.hidden_size = hidden_size

        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.candidate = nn.Linear(input_size + hidden_size, hidden_size)

        self.hidden2out = nn.Linear(hidden_size, output_size)

    def name(self):
        return "GRU"

    def forward(self, x, hidden_state):
        combined = torch.cat((x, hidden_state), dim=1)

        update = torch.sigmoid(self.update_gate(combined))
        reset = torch.sigmoid(self.reset_gate(combined))

        reset_hidden = reset * hidden_state
        candidate_combined = torch.cat((x, reset_hidden), dim=1)
        candidate_hidden = torch.tanh(self.candidate(candidate_combined))

        hidden = (1 - update) * hidden_state + update * candidate_hidden
        output = self.hidden2out(hidden)

        return output, hidden

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size, device=next(self.parameters()).device)

class ExMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExMLP, self).__init__()

        self.ReLU = torch.nn.ReLU()

        self.layer1 = MatMul(input_size, hidden_size)
        self.layer2 = MatMul(hidden_size, hidden_size)
        self.layer3 = MatMul(hidden_size, output_size)

    def name(self):
        return "MLP"

    def forward(self, x):
        x = self.layer1(x)
        x = self.ReLU(x)

        x = self.layer2(x)
        x = self.ReLU(x)

        x = self.layer3(x)

        return x

class ExLRestSelfAtten(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExLRestSelfAtten, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.atten_size = atten_size
        self.window_size = 2 * atten_size + 1

        self.sqrt_hidden_size = np.sqrt(float(hidden_size))
        self.ReLU = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=2)

        # First token-wise FC layer: word embedding -> hidden representation
        self.layer1 = MatMul(input_size, hidden_size)

        # Learnable local positional encoding for offsets [-atten_size, ..., +atten_size]
        self.pos_encoding = nn.Parameter(
            0.01 * torch.randn(1, 1, self.window_size, hidden_size)
        )

        # Single-head self-attention matrices
        self.W_q = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_k = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_v = MatMul(hidden_size, hidden_size, use_bias=False)

        # Token-wise classifier after contextualization
        self.layer2 = MatMul(hidden_size, hidden_size)
        self.layer3 = MatMul(hidden_size, output_size)

    def name(self):
        return "MLP_atten"

    def forward(self, x):
        # x shape: [batch_size, num_words, input_size]

        x = self.layer1(x)
        x = self.ReLU(x)
        # x shape: [batch_size, num_words, hidden_size]

        # Create local neighborhoods using zero padding and torch.roll.
        # x_nei shape: [batch_size, num_words, window_size, hidden_size]
        padded = pad(x, (0, 0, self.atten_size, self.atten_size, 0, 0))

        x_nei = []
        for k in range(-self.atten_size, self.atten_size + 1):
            x_nei.append(torch.roll(padded, k, dims=1))

        x_nei = torch.stack(x_nei, dim=2)
        x_nei = x_nei[:, self.atten_size:-self.atten_size, :, :]

        # Add local positional encoding to the neighborhood vectors.
        x_nei = x_nei + self.pos_encoding

        # Single-head attention:
        # query shape: [batch_size, num_words, 1, hidden_size]
        # keys/vals shape: [batch_size, num_words, window_size, hidden_size]
        query = self.W_q(x).unsqueeze(2)
        keys = self.W_k(x_nei)
        vals = self.W_v(x_nei)

        # Attention scores over the local window.
        # atten_scores shape: [batch_size, num_words, window_size]
        atten_scores = torch.sum(query * keys, dim=-1) / self.sqrt_hidden_size
        atten_weights = self.softmax(atten_scores)

        # Weighted average of local values.
        # x shape: [batch_size, num_words, hidden_size]
        x = torch.sum(atten_weights.unsqueeze(-1) * vals, dim=2)

        # Token-wise sub prediction scores.
        x = self.layer2(x)
        x = self.ReLU(x)
        x = self.layer3(x)

        # x shape: [batch_size, num_words, output_size]
        return x, atten_weights

# prints portion of the review (20-30 first words), with the sub-scores each work obtained
# prints also the final scores, the softmaxed prediction values and the true label values

def print_review(rev_text, sbs1, sbs2, lbl1, lbl2):
    print("\nReview words and sub-prediction scores:")
    print(f"True label vector: positive={lbl1:.1f}, negative={lbl2:.1f}")
    print(f"{'word':<20} {'pos_score':>12} {'neg_score':>12}")

    max_words_to_print = min(len(rev_text), 30)

    for i in range(max_words_to_print):
        print(f"{rev_text[i]:<20} {sbs1[i]:>12.4f} {sbs2[i]:>12.4f}")

    final_pos = float(np.mean(sbs1[:len(rev_text)]))
    final_neg = float(np.mean(sbs2[:len(rev_text)]))

    probs = torch.softmax(torch.tensor([final_pos, final_neg]), dim=0)

    print(f"Final averaged logits: positive={final_pos:.4f}, negative={final_neg:.4f}")
    print(f"Softmax probabilities: positive={float(probs[0]):.4f}, negative={float(probs[1]):.4f}")


# select model to use

if run_recurrent:
    if use_RNN:
        model = ExRNN(input_size, output_size, hidden_size)
    else:
        model = ExGRU(input_size, output_size, hidden_size)
else:
    if atten_size > 0:
        model = ExLRestSelfAtten(input_size, output_size, hidden_size)
    else:
        model = ExMLP(input_size, output_size, hidden_size)

print("Using model: " + model.name())

model = model.to(ld.device)

if reload_model:
    print("Reloading model")
    model.load_state_dict(torch.load(model.name() + ".pth"))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def evaluate_full(model, data_loader, criterion):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for labels, reviews, reviews_text in data_loader:
            target_labels = torch.argmax(labels, dim=1) if labels.dim() == 2 else labels
            target_labels = target_labels.long()

            if run_recurrent:
                hidden_state = model.init_hidden(int(labels.shape[0]))

                for i in range(num_words):
                    output, hidden_state = model(reviews[:, i, :], hidden_state)

            else:
                if atten_size > 0:
                    sub_score, atten_weights = model(reviews)
                else:
                    sub_score = model(reviews)

                mask = (reviews.abs().sum(dim=-1) > 0).float()
                expanded_mask = mask.unsqueeze(-1)

                masked_sub_scores = sub_score * expanded_mask
                output = masked_sub_scores.sum(dim=1) / mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
            loss = criterion(output, target_labels)

            batch_size_actual = int(labels.shape[0])
            predicted = torch.argmax(output, dim=1)

            total_loss += float(loss.detach()) * batch_size_actual
            total_correct += int((predicted == target_labels).sum())
            total_samples += batch_size_actual

    model.train()

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc

train_loss = 1.0
test_loss = 1.0
train_acc = 0.0
test_acc = 0.0

history_epochs = []
history_train_loss = []
history_test_loss = []
history_train_acc = []
history_test_acc = []

global_step = 0

test_iterator = iter(test_dataset)

# training steps in which a test step is executed every test_interval

for epoch in range(num_epochs):

    itr = 0  # iteration counter within each epoch

    for labels, reviews, reviews_text in train_dataset:  # getting training batches

        itr = itr + 1
        global_step = global_step + 1

        if (itr + 1) % test_interval == 0:
            test_iter = True

            # Safely grab the next test batch
            try:
                labels, reviews, reviews_text = next(test_iterator)
            except StopIteration:
                # If we run out of test data, reset the iterator!
                test_iterator = iter(test_dataset)
                labels, reviews, reviews_text = next(test_iterator)

        else:
            test_iter = False

        target_labels = torch.argmax(labels,
                                     dim=1) if labels.dim() == 2 else labels
        target_labels = target_labels.long()

        # Recurrent nets (RNN/GRU)

        if run_recurrent:
            hidden_state = model.init_hidden(int(labels.shape[0]))

            for i in range(num_words):
                output, hidden_state = model(reviews[:, i, :], hidden_state)

        else:

            # Token-wise networks (MLP / MLP + Atten.)

            sub_score = []
            if atten_size > 0:
                # MLP + atten
                sub_score, atten_weights = model(reviews)
            else:
                # MLP
                sub_score = model(reviews)

            # Creates a mask of shape [batch_size, num_words] (1 for real words, 0 for padding)
            mask = (reviews.abs().sum(dim=-1) > 0).float()

            # sub_score shape: [batch_size, num_words, 2]
            # Expand mask to match sub_score shape: [batch_size, num_words, 1]
            expanded_mask = mask.unsqueeze(-1)

            # Zero out padded sub-scores
            masked_sub_scores = sub_score * expanded_mask

            # Sum the scores and divide by the actual number of words (sum of the mask)
            output = masked_sub_scores.sum(dim=1) / mask.sum(dim=1).clamp(min=1).unsqueeze(-1)

        # cross-entropy loss
        loss = criterion(output, target_labels)

        # accuracy
        predicted = torch.argmax(output, dim=1)
        accuracy = (predicted == target_labels).float().mean()

        # optimize in training iterations

        if not test_iter:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # averaged losses and accuracies
        if test_iter:
            test_loss = 0.8 * float(loss.detach()) + 0.2 * test_loss
            test_acc = 0.8 * float(accuracy.detach()) + 0.2 * test_acc
        else:
            train_loss = 0.9 * float(loss.detach()) + 0.1 * train_loss
            train_acc = 0.9 * float(accuracy.detach()) + 0.1 * train_acc

        if test_iter:

            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Step [{itr + 1}/{len(train_dataset)}], "
                f"Train Loss: {train_loss:.4f}, "
                f"Test Loss: {test_loss:.4f}, "
                f"Train Acc: {train_acc:.4f}, "
                f"Test Acc: {test_acc:.4f}"
            )

            # Per-word sub-score analysis is done after training on custom reviews.

            # saving the model
            torch.save(model, f"{model.name()}_hidden_{hidden_size}.pth")

    full_train_loss, full_train_acc = evaluate_full(model, train_dataset,
                                                    criterion)
    full_test_loss, full_test_acc = evaluate_full(model, test_dataset,
                                                  criterion)

    history_epochs.append(epoch + 1)
    history_train_loss.append(full_train_loss)
    history_test_loss.append(full_test_loss)
    history_train_acc.append(full_train_acc)
    history_test_acc.append(full_test_acc)

    print("\nFull evaluation after epoch:")
    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    print(
        f"Train Loss: {full_train_loss:.4f}, Train Acc: {full_train_acc:.4f}")
    print(f"Test Loss:  {full_test_loss:.4f}, Test Acc:  {full_test_acc:.4f}")

full_train_loss, full_train_acc = evaluate_full(model, train_dataset,
                                                criterion)
full_test_loss, full_test_acc = evaluate_full(model, test_dataset, criterion)

print("\nFull evaluation after training:")
print(f"Train Loss: {full_train_loss:.4f}, Train Acc: {full_train_acc:.4f}")
print(f"Test Loss:  {full_test_loss:.4f}, Test Acc:  {full_test_acc:.4f}")

def predict_mlp_review(model, review_text):
    model.eval()

    review_tensor = ld.preprocess_review(review_text).to(ld.device)
    words = ld.tokinize(review_text)

    with torch.no_grad():
        if atten_size > 0:
            sub_score, atten_weights = model(review_tensor)
        else:
            sub_score = model(review_tensor)

        mask = (review_tensor.abs().sum(dim=-1) > 0).float()
        output = (
            (sub_score * mask.unsqueeze(-1)).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        )

        probs = torch.softmax(output, dim=1).squeeze(0)

    positive_prob = float(probs[0])
    negative_prob = float(probs[1])
    predicted_label = "positive" if positive_prob > negative_prob else "negative"

    return (
        predicted_label,
        positive_prob,
        negative_prob,
        words,
        sub_score.squeeze(0).detach().cpu().numpy()
    )

if not run_recurrent:
    custom_reviews = [
        ("positive", "This movie was wonderful and exciting with great acting and a beautiful story"),
        ("negative", "This movie was boring slow confusing and disappointing with terrible acting"),
        ("negative", "The actors were great and the music was beautiful but the movie was boring and disappointing"),
        ("positive", "The beginning was slow and boring but later the story became moving beautiful and excellent"),
    ]

    print("\nCustom review analysis:")

    for true_label, review_text in custom_reviews:
        pred_label, pos_prob, neg_prob, words, sub_scores = predict_mlp_review(model, review_text)

        print("\nReview:")
        print(review_text)
        print(f"True label: {true_label}")
        print(f"Predicted label: {pred_label}")
        print(f"positive={pos_prob:.4f}, negative={neg_prob:.4f}")

        if true_label == "positive" and pred_label == "positive":
            scenario = "TP"
        elif true_label == "negative" and pred_label == "negative":
            scenario = "TN"
        elif true_label == "negative" and pred_label == "positive":
            scenario = "FP"
        else:
            scenario = "FN"

        print(f"Scenario: {scenario}")
        print(f"{'word':<20} {'pos_score':>12} {'neg_score':>12}")

        for i, word in enumerate(words):
            print(f"{word:<20} {sub_scores[i, 0]:>12.4f} {sub_scores[i, 1]:>12.4f}")

plt.figure()
plt.plot(history_epochs, history_train_loss, marker="o", label="Train Loss")
plt.plot(history_epochs, history_test_loss, marker="o", label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"{model.name()} loss, hidden size = {hidden_size}")
plt.legend()
plt.grid(True)
plt.savefig(f"{model.name()}_hidden_{hidden_size}_loss.png")
plt.show()

plt.figure()
plt.plot(history_epochs, history_train_acc, marker="o", label="Train Accuracy")
plt.plot(history_epochs, history_test_acc, marker="o", label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(f"{model.name()} accuracy, hidden size = {hidden_size}")
plt.legend()
plt.grid(True)
plt.savefig(f"{model.name()}_hidden_{hidden_size}_accuracy.png")
plt.show()