"""Task 3: unpaired Monet/photo dataset, implementing the locked Policy A
epoch-length convention (epoch length = larger domain; the smaller domain
is independently resampled). Paths are supplied by the caller -- nothing
here hardcodes a personal machine path.

- Accepts .jpg/.jpeg/.png, converts to RGB.
- Resize 286x286 -> independent random 256x256 crop -> independent random
  horizontal flip, drawn separately for each domain (no shared randomness
  between A and B, since they are unpaired).
- Normalizes to [-1, 1] to match the generator's Tanh output range.
- No assumption that filenames correspond between domains.

Monet sampling (fixed from the static audit): the Monet index for a given
__getitem__ call is drawn fresh, uniformly at random, independent of the
photo index -- NOT `index % len(monet_paths)`. That modulo form made every
photo index map to the exact same Monet index in every epoch (only the
visiting order changed under shuffling), which is not true independent
resampling. Per-worker RNGs are lazily created on first access inside each
worker process (see _ensure_worker_rngs), so num_workers > 0 gives every
worker its own distinct, seed-derived, reproducible stream instead of all
workers inheriting an identical forked RNG state.
"""

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def list_images(folder):
    folder = Path(folder)
    paths = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]
    return paths


def _random_transform_params(load_size, crop_size, rng):
    max_offset = load_size - crop_size
    top = rng.randint(0, max_offset)
    left = rng.randint(0, max_offset)
    flip = rng.random() < 0.5
    return top, left, flip


def _apply_transform(image, load_size, crop_size, top, left, flip):
    image = image.resize((load_size, load_size), Image.BICUBIC)
    image = image.crop((left, top, left + crop_size, top + crop_size))
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image


def _to_tensor_normalized(image):
    # PIL RGB image -> float tensor in [-1, 1], shape (3, H, W).
    import numpy as np
    array = np.asarray(image, dtype="float32") / 255.0  # [0, 1]
    array = array * 2.0 - 1.0  # [-1, 1]
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor


class UnalignedMonetPhotoDataset(Dataset):
    """Policy A: __len__ = max(len(monet), len(photo)). The photo domain
    (the larger one) is traversed by direct index -- every photo index is
    visited exactly once per epoch (order depends on the DataLoader's
    sampler; shuffling reorders, never duplicates or omits). The Monet
    domain (the smaller one) is independently resampled: for every
    __getitem__ call, a fresh uniform-random Monet index is drawn,
    decoupled from the photo index, so the same photo index does not
    deterministically imply the same Monet index across epochs."""

    def __init__(self, monet_dir, photo_dir, load_size=286, crop_size=256, seed=None):
        self.monet_paths = list_images(monet_dir)
        self.photo_paths = list_images(photo_dir)
        if not self.monet_paths:
            raise ValueError(f"No images found in monet_dir: {monet_dir}")
        if not self.photo_paths:
            raise ValueError(f"No images found in photo_dir: {photo_dir}")

        self.load_size = load_size
        self.crop_size = crop_size
        self._base_seed = seed if seed is not None else 0

        # Lazily created per-worker-process RNGs -- see _ensure_worker_rngs.
        self._worker_rng_a = None
        self._worker_rng_b = None

        self.epoch_length = max(len(self.monet_paths), len(self.photo_paths))

    def __len__(self):
        return self.epoch_length

    def _ensure_worker_rngs(self):
        """Lazily initialize this process's RNGs on first access. With
        num_workers=0 everything runs in the main process (worker_id=0).
        With num_workers>0, each worker process gets its own copy of the
        Dataset object; calling this here (inside __getitem__, which only
        ever runs inside the worker that owns this copy) gives each worker
        a distinct, deterministic, seed-derived stream instead of every
        worker inheriting an identical forked RNG state. Note: PyTorch does
        not guarantee which worker handles which index in the same order
        across runs -- only each worker's own RNG stream is reproducible."""
        if self._worker_rng_a is not None:
            return
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        self._worker_rng_a = random.Random(self._base_seed + worker_id * 2)
        self._worker_rng_b = random.Random(self._base_seed + worker_id * 2 + 1)

    def _load_domain_image(self, paths, index, rng):
        path = paths[index % len(paths)]
        image = Image.open(path).convert("RGB")
        top, left, flip = _random_transform_params(self.load_size, self.crop_size, rng)
        image = _apply_transform(image, self.load_size, self.crop_size, top, left, flip)
        return _to_tensor_normalized(image), path.name

    def __getitem__(self, index):
        self._ensure_worker_rngs()

        # Photo (domain B, the larger domain) traversed directly by index.
        photo_tensor, photo_name = self._load_domain_image(self.photo_paths, index, self._worker_rng_b)

        # Monet (domain A, the smaller domain): independently resampled --
        # a fresh uniform-random index every call, NOT index % len(monet_paths).
        monet_index = self._worker_rng_a.randrange(len(self.monet_paths))
        monet_tensor, monet_name = self._load_domain_image(self.monet_paths, monet_index, self._worker_rng_a)

        return {
            "A": monet_tensor,
            "B": photo_tensor,
            "A_name": monet_name,
            "B_name": photo_name,
        }
