"""
Loader for the DIODE validation split (Track A).

Released layout, per frame (val/{indoors,outdoor}/scene_*/scan_*/):

    <id>.png               RGB image
    <id>_depth.npy         float32 depth in metres, shape (H, W, 1)
    <id>_depth_mask.npy    0/1 validity mask, shape (H, W)

The official validation split has 771 frames. The full run checks this
count, so a loader bug that drops frames stops the run instead of
quietly changing the benchmark.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DIODE_VAL_EXPECTED_FRAMES = 771


@dataclass
class DiodeFrame:
    image_path: Path
    depth_path: Path
    mask_path: Path
    split: str      # "indoors" or "outdoor"
    image_id: str   # path relative to the val root, without extension


def find_diode_triples(diode_val_root: str | Path) -> list[DiodeFrame]:
    root = Path(diode_val_root)
    if not root.is_dir():
        raise FileNotFoundError(f"DIODE val root not found: {root}")
    frames = []
    for depth_path in sorted(root.rglob("*_depth.npy")):
        stem = depth_path.name[: -len("_depth.npy")]
        image_path = depth_path.parent / f"{stem}.png"
        mask_path = depth_path.parent / f"{stem}_depth_mask.npy"
        if not image_path.exists() or not mask_path.exists():
            print(f"[diode_dataset] WARNING: skipping {stem}, missing image or mask")
            continue
        rel = depth_path.relative_to(root).parts
        if "indoors" in rel:
            split = "indoors"
        elif "outdoor" in rel:
            split = "outdoor"
        else:
            raise ValueError(f"Cannot tell indoor/outdoor for {depth_path}")
        image_id = str((depth_path.parent / stem).relative_to(root))
        frames.append(DiodeFrame(image_path, depth_path, mask_path, split, image_id))
    return frames


def load_depth_and_mask(frame: DiodeFrame) -> tuple[np.ndarray, np.ndarray]:
    depth = np.load(frame.depth_path).squeeze().astype(np.float64)
    mask = np.load(frame.mask_path).squeeze().astype(bool)
    # Valid = official mask AND a strictly positive depth (needed for 1/depth).
    mask &= depth > 1e-3
    return depth, mask


def sample_smoke_test_subset(frames: list[DiodeFrame], n_images: int = 20, seed: int = 0) -> list[DiodeFrame]:
    """Seeded subset with both splits represented."""
    rng = random.Random(seed)
    indoor = [f for f in frames if f.split == "indoors"]
    outdoor = [f for f in frames if f.split == "outdoor"]
    k_in = min(n_images // 2, len(indoor))
    k_out = min(n_images - k_in, len(outdoor))
    return sorted(rng.sample(indoor, k_in) + rng.sample(outdoor, k_out), key=lambda f: f.image_id)
