"""Task 2 (Step 4): finalize Khushi's Yelp Polarity preprocessing.

Builds the capped vocabulary from training tokens only, encodes every
split into fixed-length (max_sequence_length) padded/truncated integer
arrays, and persists:
  - vocab.json                    (small metadata: token_to_idx, sizes)
  - split_indices.json            (train/validation indices into the
                                    official HF train pool, + seed)
  - preprocessing_metadata.json   (counts, class balance, UNK rates,
                                    empty-review counts, config used)

The large encoded arrays themselves (train/validation/test input ids and
labels) are saved as .npy files under data_processed/ for local use by
the next step, but .npy is already gitignored repo-wide -- they are not
committed, and the raw Yelp text is never persisted here at all.
"""

import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

from split_data import load_config, stratified_split
from text_preprocessing import preprocess
from vocab import build_vocab, effective_unk_rate, encode_pad_truncate, PAD_TOKEN, UNK_TOKEN

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "task2_config.json"
DATA_DIR = Path(__file__).resolve().parent.parent / "data_processed"


def count_empty_after_preprocessing(token_lists):
    return sum(1 for tokens in token_lists if len(tokens) == 0)


def main():
    config = load_config()
    seed = config["seed"]
    train_fraction = config["split"]["train"]
    prep_cfg = config["preprocessing"]
    vocab_cap = prep_cfg["vocab_cap"]
    max_length = prep_cfg["max_sequence_length"]

    print("Loading fancyzhx/yelp_polarity ...")
    dataset = load_dataset("fancyzhx/yelp_polarity")
    train_pool = dataset["train"]
    test_split = dataset["test"]

    labels = train_pool["label"]
    train_indices, val_indices = stratified_split(labels, train_fraction, seed)
    overlap = set(train_indices) & set(val_indices)
    assert len(overlap) == 0, "Train/validation overlap detected."

    train_texts_all = train_pool["text"]
    train_labels_all = train_pool["label"]

    def subset(indices):
        return [train_texts_all[i] for i in indices], [train_labels_all[i] for i in indices]

    train_texts, train_labels = subset(train_indices)
    val_texts, val_labels = subset(val_indices)
    test_texts, test_labels = test_split["text"], test_split["label"]

    print("Preprocessing train...")
    train_tokens = [preprocess(t) for t in train_texts]
    print("Preprocessing validation...")
    val_tokens = [preprocess(t) for t in val_texts]
    print("Preprocessing test...")
    test_tokens = [preprocess(t) for t in test_texts]

    empty_train = count_empty_after_preprocessing(train_tokens)
    empty_val = count_empty_after_preprocessing(val_tokens)
    empty_test = count_empty_after_preprocessing(test_tokens)

    print(f"Empty-after-cleaning reviews: train={empty_train}, validation={empty_val}, test={empty_test}")

    print(f"Building vocabulary from TRAINING tokens only (cap={vocab_cap})...")
    token_to_idx, idx_to_token, counter = build_vocab(train_tokens, vocab_cap=vocab_cap)
    vocab_size = len(token_to_idx)
    print(f"Final vocab size (incl. <PAD>/<UNK>): {vocab_size}")

    print(f"Encoding + padding/truncating to max_sequence_length={max_length}...")
    train_ids = np.array([encode_pad_truncate(t, token_to_idx, max_length) for t in train_tokens], dtype=np.int64)
    val_ids = np.array([encode_pad_truncate(t, token_to_idx, max_length) for t in val_tokens], dtype=np.int64)
    test_ids = np.array([encode_pad_truncate(t, token_to_idx, max_length) for t in test_tokens], dtype=np.int64)

    train_labels_arr = np.array(train_labels, dtype=np.int64)
    val_labels_arr = np.array(val_labels, dtype=np.int64)
    test_labels_arr = np.array(test_labels, dtype=np.int64)

    val_unk_count, val_unk_total, val_unk_pct = effective_unk_rate(val_tokens, token_to_idx, max_length)
    test_unk_count, test_unk_total, test_unk_pct = effective_unk_rate(test_tokens, token_to_idx, max_length)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    np.save(DATA_DIR / "train_input_ids.npy", train_ids)
    np.save(DATA_DIR / "train_labels.npy", train_labels_arr)
    np.save(DATA_DIR / "validation_input_ids.npy", val_ids)
    np.save(DATA_DIR / "validation_labels.npy", val_labels_arr)
    np.save(DATA_DIR / "test_input_ids.npy", test_ids)
    np.save(DATA_DIR / "test_labels.npy", test_labels_arr)

    with open(DATA_DIR / "vocab.json", "w") as f:
        json.dump({
            "vocab_size": vocab_size,
            "vocab_cap": vocab_cap,
            "pad_token": PAD_TOKEN,
            "unk_token": UNK_TOKEN,
            "pad_id": token_to_idx[PAD_TOKEN],
            "unk_id": token_to_idx[UNK_TOKEN],
            "token_to_idx": token_to_idx,
        }, f)

    with open(DATA_DIR / "split_indices.json", "w") as f:
        json.dump({
            "seed": seed,
            "train_fraction": train_fraction,
            "train_indices": train_indices,
            "validation_indices": val_indices,
        }, f)

    metadata = {
        "dataset": config["dataset"],
        "seed": seed,
        "max_sequence_length": max_length,
        "vocab_cap": vocab_cap,
        "final_vocab_size": vocab_size,
        "counts": {
            "train": len(train_texts),
            "validation": len(val_texts),
            "test": len(test_texts),
        },
        "class_balance": {
            "train": {"label0": int(sum(1 for l in train_labels if l == 0)),
                      "label1": int(sum(1 for l in train_labels if l == 1))},
            "validation": {"label0": int(sum(1 for l in val_labels if l == 0)),
                           "label1": int(sum(1 for l in val_labels if l == 1))},
            "test": {"label0": int(sum(1 for l in test_labels if l == 0)),
                     "label1": int(sum(1 for l in test_labels if l == 1))},
        },
        "empty_after_cleaning": {
            "train": empty_train,
            "validation": empty_val,
            "test": empty_test,
        },
        "unk_rate_effective": {
            "validation": {"unk_count": val_unk_count, "total_tokens": val_unk_total, "pct": val_unk_pct},
            "test": {"unk_count": test_unk_count, "total_tokens": test_unk_total, "pct": test_unk_pct},
        },
        "encoded_shapes": {
            "train_input_ids": list(train_ids.shape),
            "validation_input_ids": list(val_ids.shape),
            "test_input_ids": list(test_ids.shape),
        },
        "train_validation_overlap": len(overlap),
    }
    with open(DATA_DIR / "preprocessing_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved to data_processed/:")
    print("  vocab.json, split_indices.json, preprocessing_metadata.json (tracked metadata)")
    print("  train/validation/test_input_ids.npy, *_labels.npy (gitignored, local use only)")

    return metadata


if __name__ == "__main__":
    main()
