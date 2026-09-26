"""Task 3: fake-image replay buffer (history of generated images), matching
the original CycleGAN/pix2pix implementation's algorithm. Two independent
instances are used -- one for fake_A, one for fake_B -- each with capacity
50, per the locked configuration.

Algorithm per queried image, once the pool is full:
  - with probability 0.5: swap the new image into a random pool slot and
    return the image that was previously stored there;
  - with probability 0.5: return the new image itself, unchanged, without
    storing it.
Before the pool is full, every queried image is simply stored and returned
as-is.
"""

import random

import torch


class ImagePool:
    def __init__(self, pool_size=50, seed=None):
        self.pool_size = pool_size
        self.images = []
        self.rng = random.Random(seed)

    def query(self, images):
        """images: tensor of shape (B, C, H, W). Returns a tensor of the
        same shape, with each element independently possibly swapped in
        from the pool's history."""
        if self.pool_size == 0:
            return images

        return_images = []
        for i in range(images.size(0)):
            image = images[i:i + 1]
            if len(self.images) < self.pool_size:
                self.images.append(image.clone())
                return_images.append(image)
            elif self.rng.random() < 0.5:
                idx = self.rng.randint(0, self.pool_size - 1)
                stored = self.images[idx].clone()
                self.images[idx] = image.clone()
                return_images.append(stored)
            else:
                return_images.append(image)
        return torch.cat(return_images, dim=0)


if __name__ == "__main__":
    # Deterministic smoke test under seed 42: verify the pool fills to
    # capacity, then that querying with a fixed seed is reproducible.
    torch.manual_seed(42)

    pool_a = ImagePool(pool_size=50, seed=42)
    pool_b = ImagePool(pool_size=50, seed=43)  # independent seed for the B pool

    # Fill past capacity with distinguishable "id" images (constant-valued).
    for i in range(75):
        img = torch.full((1, 3, 4, 4), float(i))
        out = pool_a.query(img)
        if i < 50:
            assert torch.equal(out, img), f"Pre-fill query should return the input unchanged (step {i})."

    assert len(pool_a.images) == 50, f"Pool should be capped at 50, got {len(pool_a.images)}"
    print(f"[PASS] fake_A pool fills to capacity 50 and stays capped after 75 queries.")

    # Reproducibility: two freshly-seeded pools given the same query sequence
    # must produce identical outputs.
    def run_sequence(seed):
        pool = ImagePool(pool_size=50, seed=seed)
        outputs = []
        for i in range(75):
            img = torch.full((1, 3, 4, 4), float(i))
            outputs.append(pool.query(img).clone())
        return outputs

    seq_1 = run_sequence(42)
    seq_2 = run_sequence(42)
    assert all(torch.equal(a, b) for a, b in zip(seq_1, seq_2)), "Same seed should reproduce identical query outputs."
    print("[PASS] Same-seed pools reproduce identical outputs across independent runs.")

    seq_3 = run_sequence(43)
    differs = any(not torch.equal(a, b) for a, b in zip(seq_1, seq_3))
    assert differs, "Different seeds should (almost certainly) diverge in swap behavior."
    print("[PASS] Different-seed pools diverge, confirming the RNG is actually driving swap decisions.")

    print("\nfake_A and fake_B pools should be constructed with different seeds (independent streams),")
    print("e.g. ImagePool(pool_size=50, seed=42) and ImagePool(pool_size=50, seed=43).")
