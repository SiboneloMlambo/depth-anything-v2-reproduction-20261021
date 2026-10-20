"""
Loader for the DA-2K benchmark's annotations.json.

Format (from the official DA-2K.md in the Depth-Anything-V2 repository):

{
  "image_path": [
    {"point1": [h1, w1], "point2": [h2, w2], "closer_point": "point1"},
    ...
  ],
  ...
}

Points are (row, column). DA-2K.md notes that point1 is always the
closer one; the loader keeps closer_point as given rather than relying on
that, so a label change cannot silently go unnoticed.

Scene type is not stored in the JSON. It is read from the image path,
which contains one of the eight scenario names. Any image whose scene
cannot be identified stops the run (unless explicitly allowed), so all
eight per-scenario numbers really cover every image.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

SCENE_TYPES = (
    "indoor", "outdoor", "non_real", "transparent_reflective",
    "adverse_style", "aerial", "underwater", "object",
)


@dataclass
class DA2KAnnotation:
    image_path: str
    scene_type: str
    point1: tuple[int, int]
    point2: tuple[int, int]
    closer_point: str


def infer_scene_type(image_path: str) -> str:
    for part in Path(image_path).parts:
        if part in SCENE_TYPES:
            return part
    return "unknown"


def load_da2k_annotations(annotations_json_path: str | Path, allow_unknown_scenes: bool = False) -> list[DA2KAnnotation]:
    with open(annotations_json_path) as f:
        raw = json.load(f)

    out, unknown = [], []
    for image_path, pairs in raw.items():
        scene_type = infer_scene_type(image_path)
        if scene_type == "unknown":
            unknown.append(image_path)
        for pair in pairs:
            if pair["closer_point"] not in ("point1", "point2"):
                raise ValueError(f"Unexpected closer_point {pair['closer_point']!r} in {image_path}")
            out.append(DA2KAnnotation(
                image_path=image_path,
                scene_type=scene_type,
                point1=tuple(int(v) for v in pair["point1"]),
                point2=tuple(int(v) for v in pair["point2"]),
                closer_point=pair["closer_point"],
            ))

    if unknown and not allow_unknown_scenes:
        raise ValueError(
            f"{len(unknown)} DA-2K images have no recognisable scene type in their path "
            f"(first: {unknown[:3]}). Fix infer_scene_type and log the change in run_log.md."
        )
    return out


def check_images_exist(annotations: list[DA2KAnnotation], da2k_root: Path) -> list[str]:
    """Return image keys that cannot be found under da2k_root."""
    return sorted({a.image_path for a in annotations if not (da2k_root / a.image_path).is_file()})


def sample_smoke_test_subset(annotations: list[DA2KAnnotation], n_images: int = 20, seed: int = 0) -> list[DA2KAnnotation]:
    """All pairs belonging to a seeded random subset of n_images distinct images."""
    rng = random.Random(seed)
    image_paths = sorted(set(a.image_path for a in annotations))
    chosen = set(rng.sample(image_paths, min(n_images, len(image_paths))))
    return [a for a in annotations if a.image_path in chosen]
