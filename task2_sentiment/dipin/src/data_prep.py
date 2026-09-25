"""Task 2 (Dipin): Yelp Polarity loading, analysis, preprocessing, vocab and encoding.

Pipeline per review:
  unescape literal \\n -> lowercase -> expand negation contractions -> strip
  punctuation/special chars -> whitespace tokenize -> drop stopwords (keeping
  negations) -> WordNet lemmatize (noun pass, then verb pass).

The vocabulary is built on the training split only. Sequences longer than
max_len keep the head and the last `tail` tokens, because reviews often end
with the overall verdict ("never coming back").
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = TASK_DIR / "configs" / "task2_config.json"
DATA_DIR = TASK_DIR / "data_processed"

NEGATION_CONTRACTIONS = {
    "can't": "can not", "cannot": "can not", "won't": "will not", "shan't": "shall not",
    "don't": "do not", "doesn't": "does not", "didn't": "did not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not", "haven't": "have not",
    "hasn't": "has not", "hadn't": "had not", "wouldn't": "would not", "shouldn't": "should not",
    "couldn't": "could not", "mustn't": "must not", "mightn't": "might not", "needn't": "need not",
    "ain't": "is not",
}
_CONTRACTION_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in NEGATION_CONTRACTIONS) + r")\b")
_ESCAPE_RE = re.compile(r"\\n|\\r|\\t|\\\"")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return json.load(f)


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_nltk():
    import nltk
    for pkg, path in [("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet"), ("omw-1.4", "corpora/omw-1.4")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


def load_raw(dataset_name):
    from datasets import load_dataset
    ds = load_dataset(dataset_name)
    return (list(ds["train"]["text"]), np.array(ds["train"]["label"], dtype=np.int64),
            list(ds["test"]["text"]), np.array(ds["test"]["label"], dtype=np.int64))


def stratified_split(labels, train_fraction, seed):
    """Identical algorithm to Khushi's split_data.py so both members share one validation set."""
    by_class = {}
    for i, y in enumerate(labels.tolist()):
        by_class.setdefault(y, []).append(i)
    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for y in sorted(by_class):
        idx = by_class[y][:]
        rng.shuffle(idx)
        cut = int(round(len(idx) * train_fraction))
        train_idx.extend(idx[:cut])
        val_idx.extend(idx[cut:])
    return np.array(train_idx), np.array(val_idx)


# --------------------------------------------------------------------------
# Data quality / analysis
# --------------------------------------------------------------------------

def audit_raw(texts, labels):
    """Missing / malformed entry audit on the raw split."""
    n = len(texts)
    none_or_nan = sum(1 for t in texts if t is None or (isinstance(t, float) and np.isnan(t)))
    empty = sum(1 for t in texts if isinstance(t, str) and not t.strip())
    has_escape = sum(1 for t in texts if isinstance(t, str) and "\\n" in t)
    has_html = sum(1 for t in texts if isinstance(t, str) and re.search(r"<[a-z/][^>]*>", t))
    non_ascii = sum(1 for t in texts if isinstance(t, str) and any(ord(c) > 127 for c in t))
    counts = Counter(texts)
    duplicates = sum(c - 1 for c in counts.values() if c > 1)
    bad_labels = int(np.sum(~np.isin(labels, [0, 1])))
    return {"n": n, "missing": none_or_nan, "empty_or_whitespace": empty, "bad_labels": bad_labels,
            "literal_escape_sequences": has_escape, "html_tags": has_html, "non_ascii": non_ascii,
            "exact_duplicate_texts": duplicates}


# --------------------------------------------------------------------------
# Text preprocessing
# --------------------------------------------------------------------------

class Preprocessor:
    def __init__(self, cfg):
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        ensure_nltk()
        self.cfg = cfg
        self.stopwords = set(stopwords.words("english")) - set(cfg["negation_exceptions"])
        self.lemmatizer = WordNetLemmatizer()
        self._lemma_cache = {}

    def clean(self, text):
        text = _ESCAPE_RE.sub(" ", text or "")
        text = text.lower()
        text = _CONTRACTION_RE.sub(lambda m: NEGATION_CONTRACTIONS[m.group(0)], text)
        text = _NON_ALNUM_RE.sub(" ", text)
        return _WS_RE.sub(" ", text).strip()

    def lemma(self, tok):
        # Cached per unique token: lemmatizing 560k reviews token-by-token is otherwise too slow.
        out = self._lemma_cache.get(tok)
        if out is None:
            out = self.lemmatizer.lemmatize(self.lemmatizer.lemmatize(tok, "n"), "v")
            self._lemma_cache[tok] = out
        return out

    def __call__(self, text):
        toks = [t for t in self.clean(text).split() if t not in self.stopwords]
        return [self.lemma(t) for t in toks]


def build_vocab(token_lists, min_freq, cap):
    counts = Counter()
    for toks in token_lists:
        counts.update(toks)
    words = [w for w, c in counts.most_common() if c >= min_freq][:cap]
    itos = ["<PAD>", "<UNK>"] + words
    return {w: i for i, w in enumerate(itos)}, counts


def encode(token_lists, stoi, max_len, keep_tail):
    """Returns (ids[N, max_len], true_lengths[N], unk_counts[N], truncated[N])."""
    n = len(token_lists)
    ids = np.zeros((n, max_len), dtype=np.int32)
    lengths = np.zeros(n, dtype=np.int64)
    unks = np.zeros(n, dtype=np.int64)
    truncated = np.zeros(n, dtype=bool)
    for i, toks in enumerate(token_lists):
        seq = [stoi.get(t, 1) for t in toks]
        unks[i] = sum(1 for s in seq if s == 1)
        if len(seq) > max_len:
            truncated[i] = True
            seq = seq[: max_len - keep_tail] + seq[-keep_tail:]
        ids[i, : len(seq)] = seq
        lengths[i] = len(seq)
    return ids, lengths, unks, truncated


def prepare_all(cfg, max_train=None, verbose=True, out_dir=DATA_DIR):
    """Full preprocessing. Returns a dict of arrays + metadata and writes them to data_processed/.

    max_train (smoke tests only) subsamples train/val/test to keep a local run fast.
    """
    pcfg = cfg["preprocessing"]
    train_text, train_y, test_text, test_y = load_raw(cfg["dataset"])
    train_idx, val_idx = stratified_split(train_y, cfg["split"]["train"], cfg["seed"])
    if max_train:
        rng = np.random.default_rng(cfg["seed"])
        train_idx = rng.choice(train_idx, max_train, replace=False)
        val_idx = rng.choice(val_idx, max_train // 5, replace=False)
        test_sel = rng.choice(len(test_y), max_train // 5, replace=False)
        test_text = [test_text[i] for i in test_sel]
        test_y = test_y[test_sel]

    splits = {
        "train": ([train_text[i] for i in train_idx], train_y[train_idx]),
        "validation": ([train_text[i] for i in val_idx], train_y[val_idx]),
        "test": (test_text, test_y),
    }
    pre = Preprocessor(pcfg)
    tokens = {}
    for name, (texts, _) in splits.items():
        tokens[name] = [pre(t) for t in texts]
        if verbose:
            print(f"preprocessed {name}: {len(texts)} reviews")

    stoi, counts = build_vocab(tokens["train"], pcfg["min_token_freq"], pcfg["vocab_cap"])
    out = {"stoi": stoi, "raw_text": {}, "labels": {}, "ids": {}, "lengths": {}, "unks": {}, "truncated": {}}
    meta = {"dataset": cfg["dataset"], "seed": cfg["seed"], "vocab_size": len(stoi),
            "train_unique_tokens": len(counts), "max_sequence_length": pcfg["max_sequence_length"],
            "smoke_subset": max_train, "splits": {}}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (texts, y) in splits.items():
        ids, lengths, unks, trunc = encode(tokens[name], stoi, pcfg["max_sequence_length"], pcfg["head_tail_keep_tail"])
        proc_len = np.array([len(t) for t in tokens[name]])
        out["raw_text"][name], out["labels"][name] = texts, y
        out["ids"][name], out["lengths"][name], out["unks"][name], out["truncated"][name] = ids, lengths, unks, trunc
        meta["splits"][name] = {
            "n": len(y), "label0": int((y == 0).sum()), "label1": int((y == 1).sum()),
            "empty_after_cleaning": int((proc_len == 0).sum()),
            "truncated": int(trunc.sum()), "truncated_pct": float(100 * trunc.mean()),
            "processed_length_percentiles_50_90_95_99": np.percentile(proc_len, [50, 90, 95, 99]).tolist(),
            "unk_rate_pct": float(100 * unks.sum() / max(1, proc_len.sum())),
        }
        np.save(out_dir / f"{name}_ids.npy", ids)
        np.save(out_dir / f"{name}_labels.npy", y)
        np.save(out_dir / f"{name}_lengths.npy", lengths)
    meta["train_validation_overlap"] = int(len(set(train_idx.tolist()) & set(val_idx.tolist())))
    with open(out_dir / "vocab.json", "w") as f:
        json.dump(list(stoi), f)
    with open(out_dir / "preprocessing_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(out_dir / "split_indices.json", "w") as f:
        json.dump({"seed": cfg["seed"], "train_indices": train_idx.tolist(), "validation_indices": val_idx.tolist()}, f)
    out["meta"] = meta
    out["tokens_example"] = {name: tokens[name][:5] for name in tokens}
    return out
