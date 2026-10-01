"""Task 1 (Dipin): TinyStories loading, story split, character tokenizer and fixed-length sequences.

Split: 100,000 training and 10,000 validation stories, sampled without overlap from the
HF train split with this member's own seed.

Tokenizer: character level, built from the training stories only. Characters seen fewer
than `min_char_count` times map to <unk>; <eos> marks the end of each story.

Sequences: every split is one long stream (story, <eos>, story, <eos>, ...). An example is
a window of sequence_length + 1 ids: input = window[:-1], target = window[1:]. Windows are
taken with stride sequence_length, so each character is a prediction target exactly once.
"""

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

TASK_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = TASK_DIR.parent.parent
CONFIG_PATH = TASK_DIR / "configs" / "task1_config.json"
DATA_DIR = TASK_DIR / "data_processed"

EOS, UNK = "<eos>", "<unk>"


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return json.load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_indices(pool_size, train_size, val_size, seed):
    idx = list(range(pool_size))
    random.Random(seed).shuffle(idx)
    return idx[:train_size], idx[train_size:train_size + val_size]


def load_stories(cfg, max_stories=None):
    """Returns (train_texts, val_texts, train_idx, val_idx). max_stories shrinks both splits (smoke tests)."""
    from datasets import load_dataset
    d = cfg["data"]
    ds = load_dataset(d["dataset"], split=d["source_split"])
    train_idx, val_idx = split_indices(len(ds), d["train_size"], d["validation_size"], cfg["seed"])
    if max_stories:
        train_idx, val_idx = train_idx[:max_stories], val_idx[:max(1, max_stories // 10)]
    return ds[train_idx]["text"], ds[val_idx]["text"], train_idx, val_idx


class CharTokenizer:
    def __init__(self, char_to_idx):
        self.char_to_idx = char_to_idx
        self.idx_to_char = {i: c for c, i in char_to_idx.items()}
        self.eos_id = char_to_idx[EOS]
        self.unk_id = char_to_idx[UNK]

    @classmethod
    def build(cls, texts, min_count):
        counts = Counter()
        for t in texts:
            counts.update(t)
        chars = sorted(c for c, n in counts.items() if n >= min_count)
        return cls({c: i for i, c in enumerate([EOS, UNK] + chars)}), counts

    @property
    def vocab_size(self):
        return len(self.char_to_idx)

    def encode(self, text):
        return [self.char_to_idx.get(c, self.unk_id) for c in text]

    def decode(self, ids):
        return "".join(self.idx_to_char[int(i)] for i in ids)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"char_to_idx": self.char_to_idx,
                       "idx_to_char": {str(i): c for i, c in self.idx_to_char.items()}}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f)["char_to_idx"])


def encode_stream(texts, tok):
    """All stories of a split as one id stream, each story followed by <eos>."""
    out = []
    for t in texts:
        out.extend(tok.encode(t))
        out.append(tok.eos_id)
    return np.asarray(out, dtype=np.int16)


def num_windows(stream_len, seq_len, offset=0):
    return (stream_len - offset - 1) // seq_len


def window_starts(stream_len, seq_len, offset=0):
    return torch.arange(num_windows(stream_len, seq_len, offset)) * seq_len + offset


def get_batch(stream, starts, seq_len):
    """stream: 1-D id tensor (any device). starts: 1-D window start positions. Returns (x, y) as int64."""
    idx = starts.to(stream.device)[:, None] + torch.arange(seq_len + 1, device=stream.device)[None, :]
    chunk = stream[idx].long()
    return chunk[:, :-1], chunk[:, 1:]


def prepare(cfg, max_stories=None, out_dir=DATA_DIR, verbose=True):
    """Runs the whole preprocessing pipeline and writes its outputs. Returns a dict with the
    tokenizer, the two id streams and metadata."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = cfg["data"]
    train_texts, val_texts, train_idx, val_idx = load_stories(cfg, max_stories)
    tok, counts = CharTokenizer.build(train_texts, d["min_char_count"])
    train_stream, val_stream = encode_stream(train_texts, tok), encode_stream(val_texts, tok)
    T = d["sequence_length"]
    meta = {
        "dataset": d["dataset"], "seed": cfg["seed"], "sequence_length": T, "smoke_subset": max_stories,
        "vocab_size": tok.vocab_size, "distinct_chars_in_train": len(counts),
        "chars_mapped_to_unk": len(counts) - (tok.vocab_size - 2),
        "train_validation_story_overlap": len(set(train_idx) & set(val_idx)),
    }
    for name, texts, stream in [("train", train_texts, train_stream), ("validation", val_texts, val_stream)]:
        lens = np.array([len(t) for t in texts])
        meta[name] = {"stories": len(texts), "characters_incl_eos": int(len(stream)),
                      "sequences_per_epoch": num_windows(len(stream), T),
                      "unk_rate_pct": float(100 * (stream == tok.unk_id).mean()),
                      "story_length_chars_p50_p90_max": [float(np.percentile(lens, 50)), float(np.percentile(lens, 90)),
                                                         int(lens.max())]}
        np.save(out_dir / f"{name}_stream.npy", stream)
    tok.save(out_dir / "tokenizer.json")
    with open(out_dir / "split_indices.json", "w") as f:
        json.dump({"seed": cfg["seed"], "train_indices": train_idx, "validation_indices": val_idx}, f)
    with open(out_dir / "sequences_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    if verbose:
        print(json.dumps(meta, indent=2))
    return {"tokenizer": tok, "train": train_stream, "validation": val_stream, "meta": meta,
            "example_story": train_texts[0]}


def load_prepared(out_dir=DATA_DIR):
    out_dir = Path(out_dir)
    with open(out_dir / "sequences_metadata.json") as f:
        meta = json.load(f)
    return {"tokenizer": CharTokenizer.load(out_dir / "tokenizer.json"), "meta": meta,
            "train": np.load(out_dir / "train_stream.npy"), "validation": np.load(out_dir / "validation_stream.npy")}
