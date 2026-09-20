"""Task 2 (Step 5): Khushi's three from-scratch Yelp Polarity classifiers --
Baseline (masked mean pooling), TextCNN, and Bidirectional GRU.

Every model owns its own nn.Embedding, initialized from scratch and never
shared across models. All architecture/hyperparameter values come from
task2_config.json -- nothing here introduces a new choice.

No pretrained embeddings or pretrained language models are used anywhere.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineMeanPoolClassifier(nn.Module):
    """Trainable embedding -> PAD-masked mean pooling -> linear classifier.

    An all-PAD input (zero non-PAD tokens) produces an exact, finite zero
    pooled vector: the masked sum is zero and the divisor is clamped to a
    minimum of 1, so there is never a divide-by-zero.
    """

    def __init__(self, vocab_size, embedding_dim, pad_id, output_dim):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.classifier = nn.Linear(embedding_dim, output_dim)

    def forward(self, input_ids):
        non_pad_mask = (input_ids != self.pad_id).float().unsqueeze(-1)  # (B, T, 1)
        embedded = self.embedding(input_ids)  # (B, T, C)

        summed = (embedded * non_pad_mask).sum(dim=1)  # (B, C); PAD positions contribute 0
        non_pad_count = non_pad_mask.sum(dim=1).clamp(min=1.0)  # (B, 1); avoid divide-by-zero
        pooled = summed / non_pad_count  # all-PAD rows: 0 / 1 = exact zero vector

        return self.classifier(pooled)


class TextCNNClassifier(nn.Module):
    """Trainable embedding -> multi-kernel 1D convolutions -> convolution-
    window-aware max-over-time pooling -> dropout -> linear classifier.

    PAD handling is window-aware: a convolution window is only eligible
    for max-pooling if every position it covers is non-PAD. Windows that
    touch any PAD position are masked to -inf before pooling, so PAD can
    never create an artificial n-gram feature. If a sample has zero
    eligible windows for a given kernel size, that kernel's pooled feature
    is a finite zero vector instead of -inf.
    """

    def __init__(self, vocab_size, embedding_dim, pad_id, kernel_sizes,
                 filters_per_kernel, output_dim, dropout):
        super().__init__()
        self.pad_id = pad_id
        self.kernel_sizes = kernel_sizes
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, filters_per_kernel, kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(filters_per_kernel * len(kernel_sizes), output_dim)

    def _window_validity(self, non_pad, kernel_size):
        """non_pad: (B, 1, T) float 0/1. Returns (B, 1, T-k+1) bool -- True
        where every one of the k positions in that window is non-PAD."""
        window_counts = F.conv1d(
            non_pad, weight=torch.ones(1, 1, kernel_size, device=non_pad.device)
        )  # (B, 1, T-k+1); each entry = count of non-PAD positions in that window
        return window_counts.round() == kernel_size

    def forward(self, input_ids):
        B = input_ids.shape[0]
        non_pad = (input_ids != self.pad_id).float().unsqueeze(1)  # (B, 1, T)

        embedded = self.embedding(input_ids)  # (B, T, C)
        embedded = embedded.transpose(1, 2)  # (B, C, T) for Conv1d

        pooled_per_kernel = []
        for conv, k in zip(self.convs, self.kernel_sizes):
            conv_out = conv(embedded)  # (B, filters, T-k+1)
            conv_out = F.relu(conv_out)

            window_valid = self._window_validity(non_pad, k)  # (B, 1, T-k+1) bool
            conv_out = conv_out.masked_fill(~window_valid, float("-inf"))

            has_valid_window = window_valid.any(dim=2)  # (B, 1)
            max_val, _ = conv_out.max(dim=2)  # (B, filters); may be -inf where no valid window

            zeros = torch.zeros_like(max_val)
            pooled_k = torch.where(has_valid_window.expand_as(max_val), max_val, zeros)
            pooled_per_kernel.append(pooled_k)

        pooled = torch.cat(pooled_per_kernel, dim=1)  # (B, filters * len(kernel_sizes))
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


class BidirectionalGRUClassifier(nn.Module):
    """Trainable embedding -> bidirectional GRU (packed by true length) ->
    concatenated final forward+backward hidden state -> dropout -> linear
    classifier.

    num_layers=1, so the GRU constructor's own `dropout` argument is fixed
    at 0 (it would silently no-op between layers anyway with one layer).
    The configured dropout is applied as its own explicit nn.Dropout layer
    on the final concatenated hidden vector instead.

    Samples with zero non-PAD tokens are never passed into
    pack_padded_sequence (which requires length >= 1): they are excluded
    from the GRU call entirely and assigned a finite zero hidden
    representation directly.
    """

    def __init__(self, vocab_size, embedding_dim, pad_id, hidden_size,
                 num_layers, bidirectional, output_dim, dropout,
                 gradient_clip_max_norm=None):
        super().__init__()
        self.pad_id = pad_id
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.gradient_clip_max_norm = gradient_clip_max_norm

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.gru = nn.GRU(
            embedding_dim, hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=0.0,  # num_layers=1 -> forced to 0, external dropout applied below
        )
        rep_dim = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(rep_dim, output_dim)

    def forward(self, input_ids):
        B, T = input_ids.shape
        device = input_ids.device
        rep_dim = self.hidden_size * (2 if self.bidirectional else 1)

        lengths = (input_ids != self.pad_id).sum(dim=1)  # (B,)
        representation = torch.zeros(B, rep_dim, device=device, dtype=torch.float32)

        nonzero_mask = lengths > 0
        if nonzero_mask.any():
            valid_idx = nonzero_mask.nonzero(as_tuple=True)[0]
            valid_input = input_ids[valid_idx]
            valid_lengths = lengths[valid_idx]

            embedded = self.embedding(valid_input)  # (B_valid, T, C)
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, valid_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, h_n = self.gru(packed)  # h_n: (num_layers * num_directions, B_valid, hidden_size)

            if self.bidirectional:
                h_forward = h_n[-2]  # last layer, forward direction
                h_backward = h_n[-1]  # last layer, backward direction
                valid_rep = torch.cat([h_forward, h_backward], dim=-1)  # (B_valid, 2*hidden_size)
            else:
                valid_rep = h_n[-1]

            representation[valid_idx] = valid_rep

        representation = self.dropout(representation)
        return self.classifier(representation)


def build_model(model_name, config):
    """Instantiate a model from task2_config.json['models'][model_name],
    with its own fresh embedding table (never shared across models)."""
    model_cfg = config["models"][model_name]
    vocab_size = config["_vocab_size"]  # injected by the caller from vocab.json
    pad_id = model_cfg["pad_id"]
    embedding_dim = model_cfg["embedding_dim"]
    output_dim = model_cfg["output_logits"]

    if model_name == "baseline":
        return BaselineMeanPoolClassifier(vocab_size, embedding_dim, pad_id, output_dim)

    if model_name == "experimental_1":
        return TextCNNClassifier(
            vocab_size, embedding_dim, pad_id,
            kernel_sizes=model_cfg["kernel_sizes"],
            filters_per_kernel=model_cfg["filters_per_kernel"],
            output_dim=output_dim,
            dropout=model_cfg["dropout"],
        )

    if model_name == "experimental_2":
        return BidirectionalGRUClassifier(
            vocab_size, embedding_dim, pad_id,
            hidden_size=model_cfg["hidden_size"],
            num_layers=model_cfg["num_layers"],
            bidirectional=model_cfg["bidirectional"],
            output_dim=output_dim,
            dropout=model_cfg["dropout"],
            gradient_clip_max_norm=model_cfg.get("gradient_clip_max_norm"),
        )

    raise ValueError(f"Unknown model_name: {model_name}")
