"""Task 1 (Dipin): decoder-only GPT written from scratch.

No prebuilt Transformer or attention modules are used: no nn.Transformer*, no
nn.MultiheadAttention, no F.scaled_dot_product_attention, and LayerNorm is implemented
here too. Only nn.Linear, nn.Embedding, nn.Dropout and tensor ops.

Block (pre-LayerNorm):
    x = x + Dropout(MultiHeadSelfAttention(LayerNorm(x)))
    x = x + Dropout(FeedForward(LayerNorm(x)))
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """Normalizes the last dimension to zero mean / unit variance, then applies a learned gain and bias."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.gain + self.bias


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, max_len, dropout, bias):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)  # query, key, value projections in one matrix
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.attn_drop = nn.Dropout(dropout)
        # Lower-triangular mask: position t may attend to positions <= t only.
        self.register_buffer("causal_mask", torch.tril(torch.ones(max_len, max_len, dtype=torch.bool)),
                             persistent=False)

    def forward(self, x, return_weights=False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        # (B, T, C) -> (B, heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, heads, T, T)
        scores = scores.masked_fill(~self.causal_mask[:T, :T], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = self.attn_drop(weights) @ v  # (B, heads, T, head_dim)
        out = self.proj(out.transpose(1, 2).contiguous().view(B, T, C))
        return (out, weights) if return_weights else out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, bias):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, num_heads, ff_dim, max_len, dropout, bias):
        super().__init__()
        self.ln1 = LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, max_len, dropout, bias)
        self.ln2 = LayerNorm(dim)
        self.ff = FeedForward(dim, ff_dim, bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, max_len, dim, num_heads, num_blocks, ff_dim, dropout, bias=False, tie_weights=True):
        super().__init__()
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(dim, num_heads, ff_dim, max_len, dropout, bias) for _ in range(num_blocks)])
        self.ln_f = LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init)
        # Scaled init for the layers that write into the residual stream (GPT-2 convention).
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * num_blocks))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.max_len, f"sequence length {T} exceeds context size {self.max_len}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, greedy=False, banned_ids=None, stop_id=None):
        """Autoregressive decoding. idx: (B, T) prompt ids. Greedy takes the argmax; otherwise
        samples from softmax(logits / temperature). banned_ids are never produced."""
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.max_len:])
            logits = logits[:, -1, :].float()
            if banned_ids:
                logits[:, banned_ids] = float("-inf")
            if greedy:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                nxt = torch.multinomial(F.softmax(logits / temperature, dim=-1), num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            if stop_id is not None and idx.size(0) == 1 and nxt.item() == stop_id:
                break
        return idx


def build_model(cfg, vocab_size):
    m = cfg["model"]
    return GPT(vocab_size=vocab_size, max_len=cfg["data"]["sequence_length"], dim=m["embedding_dim"],
               num_heads=m["num_heads"], num_blocks=m["num_transformer_blocks"], ff_dim=m["feed_forward_dim"],
               dropout=m["dropout"], bias=m["bias"], tie_weights=m["tie_input_output_embeddings"])


def count_parameters(model):
    """Unique trainable parameters (tied input/output embedding counted once)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
