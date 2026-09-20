"""Task 1 preprocessing (Step 2): character-level tokenizer built from
Khushi's selected TinyStories training stories."""

import json
from pathlib import Path

from datasets import load_dataset

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "task1_config.json"
SPLIT_INDICES_PATH = Path(__file__).resolve().parent.parent / "data_processed" / "split_indices.json"
TOKENIZER_PATH = Path(__file__).resolve().parent.parent / "data_processed" / "tokenizer.json"


def load_split_indices(path: Path = SPLIT_INDICES_PATH) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_train_texts(train_indices):
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    return [dataset[i]["text"] for i in train_indices]


def build_vocab(texts):
    chars = set()
    for text in texts:
        chars.update(text)
    vocab = sorted(chars)
    char_to_idx = {ch: i for i, ch in enumerate(vocab)}
    idx_to_char = {i: ch for i, ch in enumerate(vocab)}
    return vocab, char_to_idx, idx_to_char


def encode(text, char_to_idx):
    return [char_to_idx[ch] for ch in text]


def decode(indices, idx_to_char):
    return "".join(idx_to_char[i] for i in indices)


def save_tokenizer(vocab, char_to_idx, output_path: Path = TOKENIZER_PATH):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "vocab_size": len(vocab),
        "vocab": vocab,
        "char_to_idx": char_to_idx,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"Saved tokenizer metadata to {output_path}")


def main():
    split_indices = load_split_indices()
    train_indices = split_indices["train_indices"]

    print("Loading roneneldan/TinyStories (train split) for vocabulary construction...")
    train_texts = load_train_texts(train_indices)
    print(f"Loaded {len(train_texts)} training stories for vocabulary construction.")

    vocab, char_to_idx, idx_to_char = build_vocab(train_texts)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Vocabulary: {vocab}")

    # Verify round-trip on several samples, including an empty-string edge case.
    samples = [train_texts[0], train_texts[1], train_texts[-1], ""]
    all_ok = True
    for i, sample_text in enumerate(samples):
        encoded = encode(sample_text, char_to_idx)
        decoded = decode(encoded, idx_to_char)
        ok = decoded == sample_text
        all_ok = all_ok and ok
        label = "empty string" if sample_text == "" else f"sample {i}"
        print(f"Round-trip check ({label}): len={len(sample_text)} -> encoded len={len(encoded)} -> match={ok}")

    assert all_ok, "Tokenizer round-trip verification failed."
    print("All round-trip checks passed: decode(encode(text)) == text")

    save_tokenizer(vocab, char_to_idx)

    return vocab, char_to_idx, idx_to_char


if __name__ == "__main__":
    main()
