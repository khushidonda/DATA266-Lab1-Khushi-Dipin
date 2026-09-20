"""Task 2 (Step 2): reproducible stratified train/validation split of
Khushi's Yelp Polarity training pool. The official 38,000-example test
split is never touched here."""

import json
import random
from pathlib import Path

from datasets import load_dataset

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "task2_config.json"


def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return json.load(f)


def stratified_split(labels, train_fraction, seed):
    """Deterministic stratified split of the official training pool into
    train/validation index lists. Each class is shuffled and split
    independently so class balance is preserved in both subsets."""
    indices_by_class = {}
    for i, label in enumerate(labels):
        indices_by_class.setdefault(label, []).append(i)

    rng = random.Random(seed)
    train_indices = []
    val_indices = []
    for label in sorted(indices_by_class):
        shuffled = indices_by_class[label][:]
        rng.shuffle(shuffled)
        split_point = int(round(len(shuffled) * train_fraction))
        train_indices.extend(shuffled[:split_point])
        val_indices.extend(shuffled[split_point:])

    return train_indices, val_indices


def main():
    config = load_config()
    seed = config["seed"]
    if seed is None:
        raise ValueError(f"config['seed'] is null in {CONFIG_PATH}. Set a seed before splitting.")
    train_fraction = config["split"]["train"]
    if train_fraction is None:
        raise ValueError(f"config['split']['train'] is null in {CONFIG_PATH}.")

    print("Loading fancyzhx/yelp_polarity (train split, official pool)...")
    dataset = load_dataset("fancyzhx/yelp_polarity", split="train")
    labels = dataset["label"]

    train_indices, val_indices = stratified_split(labels, train_fraction, seed)
    overlap = set(train_indices) & set(val_indices)

    print(f"Seed: {seed}, train_fraction: {train_fraction}")
    print(f"Official train pool size: {len(labels)}")
    print(f"Train indices: {len(train_indices)}")
    print(f"Validation indices: {len(val_indices)}")
    print(f"Overlap: {len(overlap)}")
    assert len(overlap) == 0, "Train/validation overlap detected."

    for split_name, indices in [("train", train_indices), ("validation", val_indices)]:
        split_labels = [labels[i] for i in indices]
        n = len(split_labels)
        n_pos = sum(split_labels)
        n_neg = n - n_pos
        print(f"  {split_name}: n={n}, label0={n_neg} ({100 * n_neg / n:.2f}%), "
              f"label1={n_pos} ({100 * n_pos / n:.2f}%)")

    return train_indices, val_indices


if __name__ == "__main__":
    main()
