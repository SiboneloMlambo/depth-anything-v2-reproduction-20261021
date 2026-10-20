"""
Track A: zero-shot DIODE validation evaluation for V1-S and V2-S.

Primary protocol: per-image scale-and-shift fitted in DISPARITY space,
then inverted (src/alignment.py). The depth-space fit is computed too,
only as a sensitivity number for the write-up.

Reports indoor, outdoor and combined results separately, and V2-minus-V1
changes with paired bootstrap CIs over images.

Usage:
    python -m src.run_track_a_diode --diode-root data/diode/val \
        --checkpoint-v1 checkpoints/depth_anything_v1_vits.pth \
        --checkpoint-v2 checkpoints/depth_anything_v2_vits.pth \
        [--outdir results/diode] [--smoke-test] [--force]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.alignment import DEFAULT_MAX_DEPTH, align_depth_space, align_disparity_to_depth
from src.diode_dataset import (
    DIODE_VAL_EXPECTED_FRAMES, find_diode_triples, load_depth_and_mask, sample_smoke_test_subset,
)
from src.metrics import DiodeImageResult, abs_rel, bootstrap_ci, bootstrap_paired_difference, delta1
from src.model_loader import infer_disparity, load_depth_anything_vits

SPLITS = ("indoors", "outdoor", "combined")
MODELS = (("v1s", "v1"), ("v2s", "v2"))


def read_image(path: Path):
    import cv2

    return cv2.imread(str(path))


def resize_to(pred: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    import cv2

    return cv2.resize(pred, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def evaluate_frame(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, frame, max_depth: float) -> DiodeImageResult:
    aligned, s, t, n_clipped = align_disparity_to_depth(pred, gt, mask, max_depth=max_depth)
    result = DiodeImageResult(
        image_id=frame.image_id, split=frame.split,
        abs_rel=abs_rel(aligned, gt, mask), delta1=delta1(aligned, gt, mask),
        n_valid_px=int(mask.sum()), n_clipped_px=n_clipped, scale=s, shift=t,
    )
    # Sensitivity: depth-space fit; non-positive aligned depths are excluded and counted.
    aligned_d, _, _ = align_depth_space(pred, gt, mask)
    positive = mask & (aligned_d > 1e-6)
    result.n_nonpositive_depthfit = int(mask.sum() - positive.sum())
    if positive.sum() >= 10:
        result.abs_rel_depthfit = abs_rel(aligned_d, gt, positive)
        result.delta1_depthfit = delta1(aligned_d, gt, positive)
    return result


def evaluate_checkpoint(model, frames, input_size: int = 518, max_depth: float = DEFAULT_MAX_DEPTH):
    results, skipped = [], []
    for frame in tqdm(frames, desc="DIODE eval"):
        raw_image = read_image(frame.image_path)
        if raw_image is None:
            skipped.append((frame.image_id, "unreadable image"))
            continue
        gt, mask = load_depth_and_mask(frame)
        if mask.sum() < 10:
            skipped.append((frame.image_id, f"only {int(mask.sum())} valid pixels"))
            continue
        pred = infer_disparity(model, raw_image, input_size=input_size)
        if pred.shape != gt.shape:
            skipped.append((frame.image_id, f"resized pred {pred.shape} -> {gt.shape}"))
            pred = resize_to(pred, gt.shape)
        results.append(evaluate_frame(pred, gt, mask, frame, max_depth))
    return results, skipped


def _subset(results, split):
    return results if split == "combined" else [r for r in results if r.split == split]


def summarize_model(results: list[DiodeImageResult]) -> dict:
    out = {}
    for split in SPLITS:
        sub = _subset(results, split)
        if not sub:
            continue
        ar = bootstrap_ci([r.abs_rel for r in sub])
        d1 = bootstrap_ci([r.delta1 for r in sub])
        depthfit = [r for r in sub if r.abs_rel_depthfit is not None]
        out[split] = {
            "n_images": len(sub),
            "abs_rel_mean": ar[0], "abs_rel_ci95": [ar[1], ar[2]],
            "delta1_mean": d1[0], "delta1_ci95": [d1[1], d1[2]],
            "clipped_px_fraction": float(sum(r.n_clipped_px for r in sub) / sum(r.n_valid_px for r in sub)),
            "sensitivity_depth_space_fit": {
                "n_images": len(depthfit),
                "abs_rel_mean": float(np.mean([r.abs_rel_depthfit for r in depthfit])) if depthfit else None,
                "delta1_mean": float(np.mean([r.delta1_depthfit for r in depthfit])) if depthfit else None,
                "nonpositive_px_fraction": float(
                    sum(r.n_nonpositive_depthfit for r in sub) / sum(r.n_valid_px for r in sub)),
            },
        }
    return out


def paired_comparison(v1: list[DiodeImageResult], v2: list[DiodeImageResult]) -> dict:
    a = {r.image_id: r for r in v1}
    b = {r.image_id: r for r in v2}
    common = sorted(set(a) & set(b))
    out = {"n_unpaired_dropped": len(set(a) ^ set(b))}
    for split in SPLITS:
        ids = [i for i in common if split == "combined" or a[i].split == split]
        if not ids:
            continue
        d_ar = bootstrap_paired_difference([b[i].abs_rel for i in ids], [a[i].abs_rel for i in ids])
        d_d1 = bootstrap_paired_difference([b[i].delta1 for i in ids], [a[i].delta1 for i in ids])
        out[split] = {
            "n_images": len(ids),
            "delta_abs_rel": d_ar[0], "delta_abs_rel_ci95": [d_ar[1], d_ar[2]],
            "delta_delta1": d_d1[0], "delta_delta1_ci95": [d_d1[1], d_d1[2]],
        }
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--diode-root", required=True)
    parser.add_argument("--checkpoint-v1", required=True)
    parser.add_argument("--checkpoint-v2", required=True)
    parser.add_argument("--outdir", default=None, help="default: results/diode (results/smoke/diode with --smoke-test)")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--max-depth", type=float, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--smoke-test", action="store_true", help="only 20 frames (10 indoor + 10 outdoor)")
    parser.add_argument("--expected-frames", type=int, default=DIODE_VAL_EXPECTED_FRAMES,
                        help="full run stops if the loader finds a different count (0 disables)")
    parser.add_argument("--force", action="store_true", help="re-run even if results already exist")
    args = parser.parse_args(argv)

    outdir = Path(args.outdir or ("results/smoke/diode" if args.smoke_test else "results/diode"))
    if (outdir / "diode_comparison.json").is_file() and not args.force:
        print(f"[track A] {outdir}/diode_comparison.json exists; skipping (use --force to re-run).")
        return
    outdir.mkdir(parents=True, exist_ok=True)

    frames = find_diode_triples(args.diode_root)
    n_in = sum(f.split == "indoors" for f in frames)
    print(f"Found {len(frames)} DIODE val frames ({n_in} indoor, {len(frames) - n_in} outdoor)")
    if args.smoke_test:
        frames = sample_smoke_test_subset(frames, n_images=20)
        print(f"Smoke test: using {len(frames)} frames")
    elif args.expected_frames and len(frames) != args.expected_frames:
        raise SystemExit(
            f"Expected {args.expected_frames} DIODE val frames, found {len(frames)}. "
            "Check the download/loader; if the count is genuinely different, log why in "
            "run_log.md and pass --expected-frames 0."
        )

    all_results, skipped = {}, {}
    for name, version in MODELS:
        ckpt = args.checkpoint_v1 if version == "v1" else args.checkpoint_v2
        print(f"\n=== Evaluating {name} ({ckpt}) ===")
        model = load_depth_anything_vits(ckpt, model_version=version)
        results, skipped[name] = evaluate_checkpoint(model, frames, args.input_size, args.max_depth)
        all_results[name] = results
        with open(outdir / f"diode_per_image_{name}.json", "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        del model

    comparison = {
        "protocol": {
            "alignment": "least-squares scale+shift in disparity space, then 1/disp",
            "max_depth_floor_m": args.max_depth,
            "input_size": args.input_size,
            "n_frames_found": len(frames),
            "smoke_test": args.smoke_test,
        },
        "skipped_or_modified": skipped,
        "v1s": summarize_model(all_results["v1s"]),
        "v2s": summarize_model(all_results["v2s"]),
        "v2_minus_v1": paired_comparison(all_results["v1s"], all_results["v2s"]),
    }
    # Top-level fields read by the decision rule: combined split, paired.
    comparison["delta_abs_rel"] = comparison["v2_minus_v1"]["combined"]["delta_abs_rel"]
    comparison["delta_delta1"] = comparison["v2_minus_v1"]["combined"]["delta_delta1"]

    with open(outdir / "diode_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print("\n=== DIODE comparison ===")
    print(json.dumps(comparison["v2_minus_v1"], indent=2))


if __name__ == "__main__":
    main()
