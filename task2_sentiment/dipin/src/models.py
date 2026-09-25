"""Task 2 (Dipin): LSTM-family sentiment classifiers, all with embeddings learned from scratch.

baseline        1-layer unidirectional LSTM, final hidden state at the true length
experimental_1  2-layer bidirectional LSTM, concat of top-layer final fwd/bwd states
experimental_2  same 2-layer BiLSTM, but additive attention pooling over all timesteps

Each step changes one thing, so the comparison isolates direction+depth (baseline -> exp1)
and pooling (exp1 -> exp2). All models output a single logit (positive-class score).
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, bidirectional,
                 embedding_dropout, dropout, inter_layer_dropout=0.0, attention_dim=None, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.emb_drop = nn.Dropout(embedding_dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=bidirectional, dropout=inter_layer_dropout if num_layers > 1 else 0.0)
        self.bidirectional = bidirectional
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.use_attention = attention_dim is not None
        if self.use_attention:
            # Bahdanau-style scoring: score_t = v^T tanh(W h_t)
            self.att_proj = nn.Linear(out_dim, attention_dim)
            self.att_v = nn.Linear(attention_dim, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, 1)
        nn.init.uniform_(self.embedding.weight, -0.05, 0.05)
        with torch.no_grad():
            self.embedding.weight[pad_id].zero_()

    def forward(self, ids, lengths, return_attention=False):
        # Empty-after-cleaning reviews have length 0; pack needs >= 1, and the PAD embedding is zero.
        lengths = lengths.clamp(min=1).cpu()
        x = self.emb_drop(self.embedding(ids))
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, (h_n, _) = self.lstm(packed)
        attn = None
        if self.use_attention:
            out, _ = pad_packed_sequence(out, batch_first=True, total_length=ids.size(1))
            scores = self.att_v(torch.tanh(self.att_proj(out))).squeeze(-1)
            mask = torch.arange(ids.size(1), device=ids.device)[None, :] < lengths.to(ids.device)[:, None]
            scores = scores.masked_fill(~mask, float("-inf"))
            attn = torch.softmax(scores, dim=1)
            rep = torch.bmm(attn.unsqueeze(1), out).squeeze(1)
        elif self.bidirectional:
            rep = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            rep = h_n[-1]
        logit = self.fc(self.drop(rep)).squeeze(-1)
        return (logit, attn) if return_attention else logit


def build_model(name, cfg, vocab_size):
    m = cfg["models"][name]
    return LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=m["embedding_dim"],
        hidden_size=m["hidden_size"],
        num_layers=m["num_layers"],
        bidirectional=m["bidirectional"],
        embedding_dropout=m["embedding_dropout"],
        dropout=m["dropout"],
        inter_layer_dropout=m.get("inter_layer_dropout", 0.0),
        attention_dim=m.get("attention_dim"),
        pad_id=cfg["preprocessing"]["pad_id"],
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
