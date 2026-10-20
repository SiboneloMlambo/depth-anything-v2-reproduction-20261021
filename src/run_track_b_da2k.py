"""
Track B: DA-2K ordinal-depth accuracy for V1-S and V2-S.

Reports each model's accuracy overall and per scenario, plus the V2-minus-V1
gain overall and per scenario, with paired cluster-bootstrap CIs that
resample whole images.

Usage:
    python -m src.run_track_b_da2k --da2k-root data/da2k \
        --checkpoint-v1 checkpoints/depth_anything_v1_vits.pth \
        --checkpoint-v2 checkpoints/depth_anything_v2_vits.pth \
        [--outdir results/da2k] [--smoke-test] [--force] [--lower-is-closer]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.da2k_dataset import SCENE_TYPES, check_images_exist, load_da2k_annotations, sample_smoke_test_subset
from src.metrics import (
    DA2KPairResult, bootstrap_clustered_accuracy, bootstrap_clustered_paired_accuracy_diff,
    da2k_pair_prediction, per_image_counts,
)
from src.model_loader import infer_disparity, load_depth_anything_vits

MODELS = (("v1s", "v1"), ("v2s", "v2"))


def read_image(path: Path):
    import cv2

    return cv2.imread(str(path))


def evaluate_checkpoint(model, annotations, da2k_root: Path, input_size: int = 518,
                        higher_is_closer: bool = True) -> list[DA2KPairResult]:
    by_image = defaultdict(list)
    for ann in annotations:
        by_image[ann.image_path].append(ann)

    results = []
    for image_path, anns in tqdm(sorted(by_image.items()), desc="DA-2K eval"):
        raw_image = read_image(da2k_root / image_path)
        if raw_image is None:
            raise FileNotFoundError(f"Could not read DA-2K image {da2k_root / image_path}")
        pred = infer_disparity(model, raw_image, input_size=input_size)
        if pred.shape[:2] != raw_image.shape[:2]:
            raise ValueError(f"Prediction {pred.shape} does not match image {raw_image.shape[:2]} for {image_path}")
        for ann in anns:
            predicted = da2k_pair_prediction(pred, ann.point1, ann.point2, higher_is_closer)
            results.append(DA2KPairResult(
                image_id=image_path, scene_type=ann.scene_type,
                correct=(predicted is not None and predicted == ann.closer_point),
                point1=ann.point1, point2=ann.point2, closer_point=ann.closer_point,
                tie=predicted is None,
            ))
    return results


def _counts_arrays(counts: dict, ids: list[str]):
    return (np.array([counts[i][0] for i in ids]), np.array([counts[i][1] for i in ids]))


def summarize_model(results: list[DA2KPairResult]) -> dict:
    counts = per_image_counts(results)
    out = {"n_pairs": len(results), "n_images": len(counts), "n_ties": int(sum(r.tie for r in results))}
    groups = {"overall": sorted(counts)}
    for scene in SCENE_TYPES:
        groups[scene] = sorted(i for i in counts if counts[i][2] == scene)
    for group, ids in groups.items():
        if not ids:
            continue
        correct, n = _counts_arrays(counts, ids)
        point, lo, hi = bootstrap_clustered_accuracy(correct, n)
        out[group] = {"n_images": len(ids), "n_pairs": int(n.sum()), "accuracy": point, "accuracy_ci95": [lo, hi]}
    return out


def paired_comparison(v1: list[DA2KPairResult], v2: list[DA2KPairResult]) -> dict:
    c1, c2 = per_image_counts(v1), per_image_counts(v2)
    if set(c1) != set(c2):
        raise ValueError("V1 and V2 were not scored on the same DA-2K images.")
    groups = {"overall": sorted(c1)}
    for scene in SCENE_TYPES:
        groups[scene] = sorted(i for i in c1 if c1[i][2] == scene)
    out = {}
    for group, ids in groups.items():
        if not ids:
            continue
        k1, n = _counts_arrays(c1, ids)
        k2, _ = _counts_arrays(c2, ids)
        point, lo, hi = bootstrap_clustered_paired_accuracy_diff(k2, k1, n)
        out[group] = {"n_images": len(ids), "n_pairs": int(n.sum()),
                      "gain_pp": 100 * point, "gain_pp_ci95": [100 * lo, 100 * hi]}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--da2k-root", required=True, help="directory containing annotations.json and images")
    parser.add_argument("--checkpoint-v1", required=True)
    parser.add_argument("--checkpoint-v2", required=True)
    parser.add_argument("--outdir", default=None, help="default: results/da2k (results/smoke/da2k with --smoke-test)")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--smoke-test", action="store_true", help="only 20 images")
    parser.add_argument("--force", action="store_true", help="re-run even if results already exist")
    parser.add_argument("--lower-is-closer", action="store_true",
                        help="flip the sign convention (only if the smoke test shows the default is wrong)")
    args = parser.parse_args(argv)

    da2k_root = Path(args.da2k_root)
    outdir = Path(args.outdir or ("results/smoke/da2k" if args.smoke_test else "results/da2k"))
    if (outdir / "da2k_comparison.json").is_file() and not args.force:
        print(f"[track B] {outdir}/da2k_comparison.json exists; skipping (use --force to re-run).")
        return
    outdir.mkdir(parents=True, exist_ok=True)

    annotations = load_da2k_annotations(da2k_root / "annotations.json")
    n_images = len(set(a.image_path for a in annotations))
    print(f"Loaded {len(annotations)} DA-2K pairs across {n_images} images")
    missing = check_images_exist(annotations, da2k_root)
    if missing:
        raise SystemExit(f"{len(missing)} annotated DA-2K images are missing under {da2k_root} (first: {missing[:3]})")
    if args.smoke_test:
        annotations = sample_smoke_test_subset(annotations, n_images=20)
        print(f"Smoke test: using {len(annotations)} pairs")

    higher_is_closer = not args.lower_is_closer
    all_results = {}
    for name, version in MODELS:
        ckpt = args.checkpoint_v1 if version == "v1" else args.checkpoint_v2
        print(f"\n=== Evaluating {name} ({ckpt}) ===")
        model = load_depth_anything_vits(ckpt, model_version=version)
        results = evaluate_checkpoint(model, annotations, da2k_root, args.input_size, higher_is_closer)
        all_results[name] = results
        with open(outdir / f"da2k_per_pair_{name}.json", "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        del model

    comparison = {
        "protocol": {"higher_is_closer": higher_is_closer, "input_size": args.input_size,
                     "tie_rule": "exact ties count as incorrect", "smoke_test": args.smoke_test,
                     "n_images_in_annotations": n_images},
        "v1s": summarize_model(all_results["v1s"]),
        "v2s": summarize_model(all_results["v2s"]),
        "v2_minus_v1": paired_comparison(all_results["v1s"], all_results["v2s"]),
    }
    comparison["v1s_accuracy"] = comparison["v1s"]["overall"]["accuracy"]
    comparison["v2s_accuracy"] = comparison["v2s"]["overall"]["accuracy"]
    comparison["delta_da2k_pp"] = comparison["v2_minus_v1"]["overall"]["gain_pp"]

    with open(outdir / "da2k_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print("\n=== DA-2K comparison ===")
    print(json.dumps(comparison["v2_minus_v1"], indent=2))


if __name__ == "__main__":
    main()
