"""
20-image smoke test on both tracks (the proposal's "Immediate milestone").

Fully automatic. Checks, in order, for BOTH checkpoints:
  1. The checkpoint loads strictly with its own release's architecture.
  2. One forward pass gives a finite map at the input image's resolution.
  3. Prediction direction: on 20 DA-2K images, ordinal accuracy under the
     default higher_is_closer convention must be >= 0.60, using DA-2K's
     own closer_point labels. A sign flip shows up as accuracy near 0.05
     instead of near 0.95, so this cannot pass by accident.
  4. DIODE alignment sanity: on 20 frames (10 indoor, 10 outdoor),
     combined AbsRel must be < 0.30 and < 5% of valid pixels may be
     clipped. (The paper's figures are ~0.07; a broken alignment gives
     ~0.8.) These bounds catch broken pipelines only. They are not
     results and play no part in the decision rule.

Writes results/smoke/smoke_report.json and exits non-zero on any failure.

Usage:
    python -m src.smoke_test --diode-root data/diode/val --da2k-root data/da2k \
        --checkpoint-v1 checkpoints/depth_anything_v1_vits.pth \
        --checkpoint-v2 checkpoints/depth_anything_v2_vits.pth
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src import run_track_a_diode, run_track_b_da2k
from src.da2k_dataset import load_da2k_annotations
from src.model_loader import infer_disparity, load_depth_anything_vits

MIN_SIGN_ACCURACY = 0.60
MAX_SMOKE_ABS_REL = 0.30
MAX_CLIPPED_FRACTION = 0.05


def check_forward_pass(checkpoint: str, version: str, image_path: Path) -> dict:
    model = load_depth_anything_vits(checkpoint, model_version=version)
    raw = run_track_b_da2k.read_image(image_path)
    pred = infer_disparity(model, raw)
    ok = bool(np.all(np.isfinite(pred)) and pred.shape[:2] == raw.shape[:2] and float(np.ptp(pred)) > 0)
    return {"loaded_strict": True, "finite_and_shaped": ok, "pred_shape": list(pred.shape),
            "image_shape": list(raw.shape[:2]), "pred_range": [float(pred.min()), float(pred.max())]}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--diode-root", required=True)
    p.add_argument("--da2k-root", required=True)
    p.add_argument("--checkpoint-v1", required=True)
    p.add_argument("--checkpoint-v2", required=True)
    p.add_argument("--outdir", default="results/smoke")
    args = p.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ckpts = ["--checkpoint-v1", args.checkpoint_v1, "--checkpoint-v2", args.checkpoint_v2]

    report, failures = {"checks": {}}, []

    # 1-2. Strict load + one forward pass, per model.
    first_image = Path(args.da2k_root) / load_da2k_annotations(Path(args.da2k_root) / "annotations.json")[0].image_path
    for name, version, ckpt in (("v1s", "v1", args.checkpoint_v1), ("v2s", "v2", args.checkpoint_v2)):
        r = check_forward_pass(ckpt, version, first_image)
        report["checks"][f"{name}_forward_pass"] = r
        if not r["finite_and_shaped"]:
            failures.append(f"{name}: forward pass not finite / wrong shape / constant")

    # 3. Prediction direction on DA-2K labels.
    run_track_b_da2k.main(["--da2k-root", args.da2k_root, *ckpts, "--smoke-test", "--force",
                           "--outdir", str(outdir / "da2k")])
    da2k = json.loads((outdir / "da2k" / "da2k_comparison.json").read_text())
    for name in ("v1s", "v2s"):
        acc = da2k[f"{name}_accuracy"]
        report["checks"][f"{name}_sign_convention_accuracy"] = acc
        if acc < MIN_SIGN_ACCURACY:
            hint = " (looks like a sign flip)" if acc < 1 - MIN_SIGN_ACCURACY else ""
            failures.append(f"{name}: smoke DA-2K accuracy {acc:.3f} < {MIN_SIGN_ACCURACY}{hint}")

    # 4. DIODE alignment sanity.
    run_track_a_diode.main(["--diode-root", args.diode_root, *ckpts, "--smoke-test", "--force",
                            "--outdir", str(outdir / "diode")])
    diode = json.loads((outdir / "diode" / "diode_comparison.json").read_text())
    for name in ("v1s", "v2s"):
        comb = diode[name]["combined"]
        report["checks"][f"{name}_diode_smoke_abs_rel"] = comb["abs_rel_mean"]
        report["checks"][f"{name}_diode_smoke_clipped_fraction"] = comb["clipped_px_fraction"]
        if not comb["abs_rel_mean"] < MAX_SMOKE_ABS_REL:
            failures.append(f"{name}: smoke DIODE AbsRel {comb['abs_rel_mean']:.3f} >= {MAX_SMOKE_ABS_REL}")
        if comb["clipped_px_fraction"] > MAX_CLIPPED_FRACTION:
            failures.append(f"{name}: {comb['clipped_px_fraction']:.1%} of DIODE pixels clipped")

    report["passed"] = not failures
    report["failures"] = failures
    (outdir / "smoke_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if failures:
        print("\nSMOKE TEST FAILED -- do not start the full run. Record this in run_log.md.", file=sys.stderr)
        sys.exit(1)
    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
