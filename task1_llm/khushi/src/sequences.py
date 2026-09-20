"""Task 1 preprocessing (Step 3): fixed-length autoregressive input/target
sequences for Khushi's train and validation TinyStories splits.

Each qualifying story is split into non-overlapping windows of
(sequence_length + 1) characters. No padding and no concatenation across
stories: stories shorter than one full window are skipped entirely, and
leftover characters at the end of a story that don't fill a full window
are dropped.
"""

import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "task1_config.json"
SPLIT_INDICES_PATH = Path(__file__).resolve().parent.parent / "data_processed" / "split_indices.json"
TOKENIZER_PATH = Path(__file__).resolve().parent.parent / "data_processed" / "tokenizer.json"
SEQUENCES_DIR = Path(__file__).resolve().parent.parent / "data_processed"
SEQUENCES_METADATA_PATH = SEQUENCES_DIR / "sequences_metadata.json"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_split_indices(path: Path = SPLIT_INDICES_PATH) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_tokenizer(path: Path = TOKENIZER_PATH):
    with open(path, "r") as f:
        payload = json.load(f)
    return payload["vocab"], payload["char_to_idx"]


def encode(text, char_to_idx):
    return [char_to_idx[ch] for ch in text]


def build_sequences(texts, char_to_idx, sequence_length):
    window_size = sequence_length + 1
    inputs = []
    targets = []
    num_skipped_stories = 0
    num_stories_used = 0

    for text in texts:
        encoded = encode(text, char_to_idx)
        if len(encoded) < window_size:
            num_skipped_stories += 1
            continue
        num_stories_used += 1
        num_chunks = len(encoded) // window_size
        for c in range(num_chunks):
            start = c * window_size
            chunk = encoded[start:start + window_size]
            inputs.append(chunk[:-1])
            targets.append(chunk[1:])

    inputs = np.array(inputs, dtype=np.int64)
    targets = np.array(targets, dtype=np.int64)
    return inputs, targets, num_stories_used, num_skipped_stories


def main():
    config = load_config()
    sequence_length = config["data"]["sequence_length"]
    if sequence_length is None:
        raise ValueError("sequence_length is null in task1_config.json.")

    split_indices = load_split_indices()
    vocab, char_to_idx = load_tokenizer()

    print("Loading roneneldan/TinyStories (train split)...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")

    train_texts = [dataset[i]["text"] for i in split_indices["train_indices"]]
    validation_texts = [dataset[i]["text"] for i in split_indices["validation_indices"]]

    window_size = sequence_length + 1
    print(f"Building sequences with sequence_length={sequence_length} (window_size={window_size})")

    train_inputs, train_targets, train_used, train_skipped = build_sequences(
        train_texts, char_to_idx, sequence_length
    )
    val_inputs, val_targets, val_used, val_skipped = build_sequences(
        validation_texts, char_to_idx, sequence_length
    )

    print(f"Train: {train_used} stories used, {train_skipped} stories skipped (< {window_size} chars)")
    print(f"Train sequences: inputs shape={train_inputs.shape}, targets shape={train_targets.shape}")
    print(f"Validation: {val_used} stories used, {val_skipped} stories skipped (< {window_size} chars)")
    print(f"Validation sequences: inputs shape={val_inputs.shape}, targets shape={val_targets.shape}")

    # Verify shift-by-one relationship holds for every sequence.
    assert np.array_equal(train_inputs[:, 1:], train_targets[:, :-1]), "Train input/target shift mismatch."
    assert np.array_equal(val_inputs[:, 1:], val_targets[:, :-1]), "Validation input/target shift mismatch."
    assert train_inputs.shape[1] == sequence_length
    assert train_targets.shape[1] == sequence_length
    assert val_inputs.shape[1] == sequence_length
    assert val_targets.shape[1] == sequence_length
    print("Verified: target == input shifted by one character for all sequences (train and validation).")

    idx_to_char = {i: ch for i, ch in enumerate(vocab)}

    def decode(indices):
        return "".join(idx_to_char[int(i)] for i in indices)

    example_input = decode(train_inputs[0])
    example_target = decode(train_targets[0])
    print("\nExample (train sequence 0):")
    print(f"  input : {example_input!r}")
    print(f"  target: {example_target!r}")

    np.save(SEQUENCES_DIR / "train_inputs.npy", train_inputs)
    np.save(SEQUENCES_DIR / "train_targets.npy", train_targets)
    np.save(SEQUENCES_DIR / "validation_inputs.npy", val_inputs)
    np.save(SEQUENCES_DIR / "validation_targets.npy", val_targets)

    metadata = {
        "sequence_length": sequence_length,
        "window_size": window_size,
        "windowing_strategy": "non_overlapping_chunks_per_story",
        "train": {
            "num_stories_used": train_used,
            "num_stories_skipped_too_short": train_skipped,
            "num_sequences": int(train_inputs.shape[0]),
        },
        "validation": {
            "num_stories_used": val_used,
            "num_stories_skipped_too_short": val_skipped,
            "num_sequences": int(val_inputs.shape[0]),
        },
    }
    with open(SEQUENCES_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved sequence metadata to {SEQUENCES_METADATA_PATH}")
    print(f"Saved sequence arrays (.npy, gitignored) to {SEQUENCES_DIR}")

    return train_inputs, train_targets, val_inputs, val_targets


if __name__ == "__main__":
    main()
