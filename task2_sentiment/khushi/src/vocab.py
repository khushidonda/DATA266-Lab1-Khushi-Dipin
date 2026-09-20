"""Task 2 (Step 4): vocabulary construction, encoding, padding, and
truncation for Khushi's Yelp Polarity pipeline. The vocabulary is built
only from training tokens -- validation and test text never influence it.

Reserved ids: 0 = <PAD>, 1 = <UNK>. Any token unseen in training, or
outside the vocabulary cap, maps to <UNK> when encoding."""

from collections import Counter

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def build_vocab(token_lists, vocab_cap=None):
    """token_lists: iterable of already-preprocessed token lists.

    If vocab_cap is given, only the top-`vocab_cap` most frequent training
    tokens (ties broken alphabetically) get their own id; everything else
    falls back to <UNK> at encode time. If vocab_cap is None, every unique
    training token gets an id (no frequency filtering)."""
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)

    ordered_tokens = sorted(counter.keys(), key=lambda t: (-counter[t], t))
    if vocab_cap is not None:
        ordered_tokens = ordered_tokens[:vocab_cap]

    token_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for i, token in enumerate(ordered_tokens, start=2):
        token_to_idx[token] = i
    idx_to_token = {i: t for t, i in token_to_idx.items()}

    return token_to_idx, idx_to_token, counter


def encode(tokens, token_to_idx):
    """Variable-length encode (no padding/truncation) -- used by the Step 2
    verification script, kept unchanged for regression comparisons."""
    unk_id = token_to_idx[UNK_TOKEN]
    return [token_to_idx.get(t, unk_id) for t in tokens]


def unk_rate(encoded_sequences, unk_id):
    """<UNK> rate over variable-length encoded sequences (no padding
    involved) -- used by the Step 2 verification script."""
    total = 0
    unk_count = 0
    for seq in encoded_sequences:
        total += len(seq)
        unk_count += sum(1 for i in seq if i == unk_id)
    pct = (100.0 * unk_count / total) if total > 0 else 0.0
    return unk_count, total, pct


def encode_pad_truncate(tokens, token_to_idx, max_length):
    """Encode a token list to ids, then truncate (keep first max_length
    tokens) or pad with <PAD> at the end so every sequence is exactly
    max_length long. An empty token list becomes an all-<PAD> sequence."""
    unk_id = token_to_idx[UNK_TOKEN]
    pad_id = token_to_idx[PAD_TOKEN]

    ids = [token_to_idx.get(t, unk_id) for t in tokens]
    ids = ids[:max_length]
    if len(ids) < max_length:
        ids = ids + [pad_id] * (max_length - len(ids))
    return ids


def effective_unk_rate(token_lists, token_to_idx, max_length):
    """<UNK> rate computed over the tokens actually seen by the model:
    each review truncated to max_length, excluding <PAD> positions."""
    unk_id = token_to_idx[UNK_TOKEN]
    total = 0
    unk_count = 0
    for tokens in token_lists:
        truncated = tokens[:max_length]
        total += len(truncated)
        unk_count += sum(1 for t in truncated if token_to_idx.get(t, unk_id) == unk_id)
    pct = (100.0 * unk_count / total) if total > 0 else 0.0
    return unk_count, total, pct
