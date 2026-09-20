"""Task 2 (Step 2): verification-only run of Khushi's Yelp Polarity
preprocessing pipeline.

Reports split counts, class balance, vocabulary size, raw->processed
examples, and <UNK> rates. Does NOT persist any large processed arrays
or vocab files to disk -- this is an in-memory check that runs before
that persistence step is approved.
"""

from datasets import load_dataset

from split_data import load_config, stratified_split
from text_preprocessing import preprocess
from vocab import build_vocab, encode, unk_rate, UNK_TOKEN


def vocab_tokens_rank(counter, token):
    """1-indexed frequency rank of a token (1 = most frequent), using the
    same ordering rule as build_vocab: count desc, then alphabetical."""
    ordered = sorted(counter.keys(), key=lambda t: (-counter[t], t))
    return ordered.index(token) + 1


def class_balance(name, labels):
    n = len(labels)
    n_pos = sum(labels)
    n_neg = n - n_pos
    print(f"[Class balance] {name}: n={n}, label0={n_neg} ({100 * n_neg / n:.2f}%), "
          f"label1={n_pos} ({100 * n_pos / n:.2f}%)")


def main():
    config = load_config()
    seed = config["seed"]
    train_fraction = config["split"]["train"]

    print("Loading fancyzhx/yelp_polarity ...")
    dataset = load_dataset("fancyzhx/yelp_polarity")
    train_pool = dataset["train"]
    test_split = dataset["test"]

    labels = train_pool["label"]
    train_indices, val_indices = stratified_split(labels, train_fraction, seed)

    overlap = set(train_indices) & set(val_indices)
    print(f"\n[Split] seed={seed}, train_fraction={train_fraction}")
    print(f"[Split] official train pool: {len(labels)}, train: {len(train_indices)}, "
          f"validation: {len(val_indices)}, test (official, untouched): {test_split.num_rows}")
    print(f"[Split] train/validation overlap: {len(overlap)}")
    assert len(overlap) == 0, "Train/validation overlap detected."

    train_texts_all = train_pool["text"]
    train_labels_all = train_pool["label"]

    def subset(indices):
        return [train_texts_all[i] for i in indices], [train_labels_all[i] for i in indices]

    train_texts, train_labels = subset(train_indices)
    val_texts, val_labels = subset(val_indices)
    test_texts, test_labels = test_split["text"], test_split["label"]

    class_balance("train", train_labels)
    class_balance("validation", val_labels)
    class_balance("test", test_labels)

    print("\nPreprocessing train texts...")
    train_tokens = [preprocess(t) for t in train_texts]
    print("Preprocessing validation texts...")
    val_tokens = [preprocess(t) for t in val_texts]
    print("Preprocessing test texts...")
    test_tokens = [preprocess(t) for t in test_texts]

    print("\nBuilding vocabulary from TRAINING tokens only...")
    token_to_idx, idx_to_token, counter = build_vocab(train_tokens)
    vocab_size = len(token_to_idx)
    print(f"[Vocab] size (including <UNK>): {vocab_size}")
    print(f"[Vocab] 15 most common tokens: {counter.most_common(15)}")

    unk_id = token_to_idx[UNK_TOKEN]
    train_encoded = [encode(t, token_to_idx) for t in train_tokens]
    val_encoded = [encode(t, token_to_idx) for t in val_tokens]
    test_encoded = [encode(t, token_to_idx) for t in test_tokens]

    train_unk_count, train_total, train_unk_pct = unk_rate(train_encoded, unk_id)
    val_unk_count, val_total, val_unk_pct = unk_rate(val_encoded, unk_id)
    test_unk_count, test_total, test_unk_pct = unk_rate(test_encoded, unk_id)

    print(f"\n[UNK rate] train:      {train_unk_count}/{train_total} tokens = {train_unk_pct:.4f}% "
          f"(expected ~0%, vocab built from train)")
    print(f"[UNK rate] validation: {val_unk_count}/{val_total} tokens = {val_unk_pct:.4f}%")
    print(f"[UNK rate] test:       {test_unk_count}/{test_total} tokens = {test_unk_pct:.4f}%")

    print("\n[Examples] raw -> processed tokens (3 per class, train split):")
    shown = {0: 0, 1: 0}
    for text, label, tokens in zip(train_texts, train_labels, train_tokens):
        if shown[label] < 3:
            print(f"  label={label}")
            print(f"    raw    : {text[:150]!r}{'...' if len(text) > 150 else ''}")
            print(f"    tokens : {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
            shown[label] += 1
        if shown[0] >= 3 and shown[1] >= 3:
            break

    print("\n[Examples] raw -> processed for reviews containing former escape artifacts "
          "(literal \\n / \\r / \\t):")
    shown_escape = 0
    for text, tokens in zip(train_texts, train_tokens):
        if ("\\n" in text or "\\r" in text or "\\t" in text) and shown_escape < 3:
            print(f"    raw    : {text[:150]!r}{'...' if len(text) > 150 else ''}")
            print(f"    tokens : {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
            shown_escape += 1

    print("\n[Artifact check] frequency/rank of token 'n' in vocabulary after the fix:")
    if "n" in counter:
        rank = vocab_tokens_rank(counter, "n")
        print(f"    'n' count={counter['n']}, rank={rank} (1 = most frequent)")
    else:
        print("    'n' does not appear in the training vocabulary at all.")

    ultra_short_count = sum(1 for t in train_texts if len(t.strip()) < 5)
    print(f"\n[Ultra-short retention] train reviews with <5 raw chars retained: {ultra_short_count}")

    print("\nVerification complete. No processed arrays or vocab files have been persisted to disk.")


if __name__ == "__main__":
    main()
