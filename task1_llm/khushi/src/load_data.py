"""Task 1 preprocessing (Step 1): load TinyStories and build Khushi's
independent, reproducible train/validation story split."""

import json
import random
from pathlib import Path

from datasets import load_dataset

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "task1_config.json"
SPLIT_INDICES_PATH = Path(__file__).resolve().parent.parent / "data_processed" / "split_indices.json"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def get_split_sizes_and_seed(config: dict):
    train_size = config["data"]["train_size"]
    validation_size = config["data"]["validation_size"]
    seed = config.get("seed")

    if seed is None:
        raise ValueError(
            f"config['seed'] is null in {CONFIG_PATH}. "
            "Set a seed before running the split."
        )

    return train_size, validation_size, seed


def build_split_indices(pool_size: int, train_size: int, validation_size: int, seed: int):
    if train_size + validation_size > pool_size:
        raise ValueError(
            f"train_size + validation_size ({train_size + validation_size}) "
            f"exceeds the available pool size ({pool_size})."
        )

    rng = random.Random(seed)
    indices = list(range(pool_size))
    rng.shuffle(indices)

    train_indices = indices[:train_size]
    validation_indices = indices[train_size:train_size + validation_size]

    return train_indices, validation_indices


def save_split_indices(train_indices, validation_indices, seed: int,
                        output_path: Path = SPLIT_INDICES_PATH):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f)
    print(f"Saved split indices to {output_path}")


def main():
    config = load_config()
    train_size, validation_size, seed = get_split_sizes_and_seed(config)

    print("Loading roneneldan/TinyStories (train split)...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    pool_size = len(dataset)
    print(f"Source pool size (HF train split): {pool_size}")

    train_indices, validation_indices = build_split_indices(
        pool_size, train_size, validation_size, seed
    )

    overlap = set(train_indices) & set(validation_indices)

    print(f"Seed used: {seed}")
    print(f"Train indices: {len(train_indices)} (expected {train_size})")
    print(f"Validation indices: {len(validation_indices)} (expected {validation_size})")
    print(f"Overlap between train/validation indices: {len(overlap)}")

    assert len(train_indices) == train_size
    assert len(validation_indices) == validation_size
    assert len(overlap) == 0

    save_split_indices(train_indices, validation_indices, seed)

    return train_indices, validation_indices


if __name__ == "__main__":
    main()
