"""Task 1 (Step 5): GPT-style character-level language model implemented
from scratch for Khushi's TinyStories subset.

All architecture values (embedding_dim, num_heads, num_transformer_blocks,
feed_forward_dim, dropout, sequence_length, vocab_size) are read from
task1_config.json and tokenizer.json — no architecture/hyperparameter
choices are made in this file.

No prebuilt Transformer/attention modules are used anywhere below
(no nn.MultiheadAttention, nn.Transformer, nn.TransformerEncoder, or any
other prebuilt attention implementation). Self-attention is implemented
manually with nn.Linear projections and explicit scaled dot-product
attention.
"""

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "task1_config.json"
TOKENIZER_PATH = Path(__file__).resolve().parent.parent / "data_processed" / "tokenizer.json"


def load_model_config():
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    with open(TOKENIZER_PATH, "r") as f:
        tokenizer = json.load(f)

    model_cfg = config["model"]
    return {
        "vocab_size": tokenizer["vocab_size"],
        "sequence_length": config["data"]["sequence_length"],
        "embedding_dim": model_cfg["embedding_dim"],
        "num_heads": model_cfg["num_heads"],
        "num_transformer_blocks": model_cfg["num_transformer_blocks"],
        "feed_forward_dim": model_cfg["feed_forward_dim"],
        "dropout": model_cfg["dropout"],
    }


class CausalSelfAttention(nn.Module):
    """Manual multi-head causal self-attention (no nn.MultiheadAttention).

    Input and output shape: (batch, seq_len, embedding_dim), written as
    (B, T, C) in the comments below.
    """

    def __init__(self, embedding_dim, num_heads, sequence_length, dropout):
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads  # size of each head's Q/K/V vectors

        # One combined projection per Q/K/V that covers all heads at once:
        # (B, T, C) -> (B, T, C), later reshaped into per-head chunks.
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Lower-triangular causal mask: position i may attend to positions <= i,
        # never to positions > i (the future). Shape (1, 1, T, T) so it
        # broadcasts over (batch, heads, T, T) attention-score tensors.
        causal_mask = torch.tril(torch.ones(sequence_length, sequence_length, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, sequence_length, sequence_length))

    def forward(self, x):
        B, T, C = x.shape  # batch, seq_len, embedding_dim

        q = self.query_proj(x)  # (B, T, C)
        k = self.key_proj(x)    # (B, T, C)
        v = self.value_proj(x)  # (B, T, C)

        # Split channel dim C into (num_heads, head_dim), then move the head
        # dimension next to batch so each head is processed independently:
        # (B, T, C) -> (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention scores for every (query, key) pair:
        # (B, nh, T, hd) @ (B, nh, hd, T) -> (B, nh, T, T).
        # Scaling by sqrt(head_dim) keeps score magnitudes stable regardless
        # of head_dim, preventing overly-confident (peaked) softmax inputs.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Block attention to future positions: set masked-out score entries
        # to -inf so softmax assigns them ~0 probability.
        scores = scores.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))

        attn = F.softmax(scores, dim=-1)  # (B, nh, T, T); each row sums to 1 over allowed (<=i) positions
        attn = self.attn_dropout(attn)

        # Weighted sum of value vectors using the attention distribution:
        # (B, nh, T, T) @ (B, nh, T, hd) -> (B, nh, T, hd)
        out = attn @ v

        # Merge heads back into a single embedding_dim channel:
        # (B, nh, T, hd) -> (B, T, nh, hd) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.output_proj(out)  # (B, T, C)
        out = self.resid_dropout(out)
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network, applied independently to every
    position: (B, T, C) -> (B, T, feed_forward_dim) -> (B, T, C)."""

    def __init__(self, embedding_dim, feed_forward_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, feed_forward_dim),
            nn.GELU(),
            nn.Linear(feed_forward_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """One decoder-only Transformer block, pre-LayerNorm design:

        x = x + SelfAttention(LayerNorm(x))
        x = x + FeedForward(LayerNorm(x))

    Both sub-layers sit inside residual ("skip") connections so gradients
    can flow directly through the block even if the sub-layer's own
    gradient is small — this is what makes stacking many blocks trainable.
    """

    def __init__(self, embedding_dim, num_heads, feed_forward_dim, sequence_length, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attn = CausalSelfAttention(embedding_dim, num_heads, sequence_length, dropout)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.ff = FeedForward(embedding_dim, feed_forward_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # (B, T, C), attention sub-layer with residual
        x = x + self.ff(self.ln2(x))    # (B, T, C), feed-forward sub-layer with residual
        return x


class GPT(nn.Module):
    """Decoder-only, GPT-style character-level language model.

    Forward pass, with B=batch size, T=sequence length, C=embedding_dim:

        token ids (B, T)
          --token_embedding-->      (B, T, C)
          --+ position_embedding--> (B, T, C)
          --N x TransformerBlock--> (B, T, C)
          --final LayerNorm-->      (B, T, C)
          --lm_head (linear)-->     (B, T, vocab_size)  logits
    """

    def __init__(self, vocab_size, sequence_length, embedding_dim, num_heads,
                 num_transformer_blocks, feed_forward_dim, dropout):
        super().__init__()
        self.sequence_length = sequence_length

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        # Learnable positional embedding: one trainable vector per absolute
        # position 0..sequence_length-1 (not a fixed sinusoidal encoding).
        self.position_embedding = nn.Embedding(sequence_length, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, feed_forward_dim, sequence_length, dropout)
            for _ in range(num_transformer_blocks)
        ])

        # Final LayerNorm before the output head: with the pre-LN design used
        # in TransformerBlock above, the residual stream itself is never
        # normalized inside the blocks, so one last LayerNorm here keeps the
        # values the lm_head sees on a consistent scale (standard GPT-2 style).
        self.final_ln = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape  # batch, seq_len
        assert T <= self.sequence_length, (
            f"sequence length {T} exceeds configured sequence_length {self.sequence_length}"
        )

        positions = torch.arange(T, device=idx.device)  # (T,)

        tok_emb = self.token_embedding(idx)            # (B, T, C)
        pos_emb = self.position_embedding(positions)    # (T, C), broadcasts over the batch dim
        x = self.embedding_dropout(tok_emb + pos_emb)    # (B, T, C)

        for block in self.blocks:
            x = block(x)  # (B, T, C)

        x = self.final_ln(x)     # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten batch and time dimensions for cross-entropy over the
            # vocabulary: (B, T, vocab_size) -> (B*T, vocab_size); (B, T) -> (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def build_model_from_config():
    cfg = load_model_config()
    model = GPT(
        vocab_size=cfg["vocab_size"],
        sequence_length=cfg["sequence_length"],
        embedding_dim=cfg["embedding_dim"],
        num_heads=cfg["num_heads"],
        num_transformer_blocks=cfg["num_transformer_blocks"],
        feed_forward_dim=cfg["feed_forward_dim"],
        dropout=cfg["dropout"],
    )
    return model, cfg
