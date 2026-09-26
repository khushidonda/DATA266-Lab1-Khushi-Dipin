"""Task 3: deterministic inference (image generation) for both CycleGAN
directions. Loads a trained checkpoint and produces:
  outputs/<run_id>/pred_A2B/   (Monet -> Photo, from G_A2B)
  outputs/<run_id>/pred_B2A/   (Photo -> Monet, from G_B2A)

Deterministic: no random crop/flip at inference -- source images are
resized directly to the target resolution. Filenames are derived from the
source image filenames (same stem, .jpg extension) so outputs are traceable
back to their inputs.

Tensor -> image conversion: [-1, 1] -> clamp -> [0, 1] -> [0, 255] uint8 ->
CHW -> HWC -> RGB image, matching the generator's Tanh output convention.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dataset import list_images
from models import build_generator
from utils import OUTPUT_DIR, load_checkpoint


def denormalize_to_image(tensor):
    """tensor: (C, H, W) float in [-1, 1] (approximately; clamped here).
    Returns a PIL RGB image."""
    tensor = tensor.detach().cpu().clamp(-1, 1)
    tensor = (tensor + 1.0) / 2.0  # [0, 1]
    array = (tensor * 255.0).round().to(torch.uint8).numpy()  # (C, H, W) uint8
    array = np.transpose(array, (1, 2, 0))  # (H, W, C)
    return Image.fromarray(array, mode="RGB")


def _load_input_tensor(path, resolution, device):
    image = Image.open(path).convert("RGB")
    image = image.resize((resolution, resolution), Image.BICUBIC)
    array = np.asarray(image, dtype="float32") / 255.0
    array = array * 2.0 - 1.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def generate_direction(generator, source_paths, output_dir, device, resolution, max_images=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = source_paths if max_images is None else source_paths[:max_images]
    generator.eval()
    written = []
    with torch.no_grad():
        for path in paths:
            input_tensor = _load_input_tensor(path, resolution, device)
            output_tensor = generator(input_tensor)[0]
            image = denormalize_to_image(output_tensor)
            out_path = output_dir / f"{path.stem}.jpg"
            image.save(out_path, quality=95)
            written.append(out_path)
    generator.train()
    return written


def generate_both_directions(checkpoint_path, monet_dir, photo_dir, run_id, resolution=256,
                              device=None, max_images_a2b=None, max_images_b2a=None,
                              output_dir=OUTPUT_DIR):
    device = device or torch.device("cpu")

    g_a2b = build_generator().to(device)
    g_b2a = build_generator().to(device)
    load_checkpoint(checkpoint_path, models={"G_A2B": g_a2b, "G_B2A": g_b2a}, map_location=device)

    monet_paths = list_images(monet_dir)
    photo_paths = list_images(photo_dir)

    pred_a2b_dir = Path(output_dir) / run_id / "pred_A2B"
    pred_b2a_dir = Path(output_dir) / run_id / "pred_B2A"

    written_a2b = generate_direction(g_a2b, monet_paths, pred_a2b_dir, device, resolution, max_images_a2b)
    written_b2a = generate_direction(g_b2a, photo_paths, pred_b2a_dir, device, resolution, max_images_b2a)

    return {
        "pred_A2B_dir": str(pred_a2b_dir),
        "pred_A2B_count": len(written_a2b),
        "pred_B2A_dir": str(pred_b2a_dir),
        "pred_B2A_count": len(written_b2a),
    }


if __name__ == "__main__":
    print("generate.py defines generation functions but does not run anything on import. "
          "Call generate_both_directions(checkpoint_path, monet_dir, photo_dir, run_id) "
          "once a real trained checkpoint exists.")
