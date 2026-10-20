"""
Scale-and-shift alignment of relative (affine-invariant) disparity to metric depth.

Depth Anything V1 and V2 (the base checkpoints, not the metric_depth
fine-tunes) output an affine-invariant *inverse depth* (disparity) map:
LARGER value = CLOSER point, with unknown scale and shift.

Because the prediction is affine in *disparity*, the scale and shift must
be fitted in disparity space, not in depth space:

    1 / gt_depth  ~=  s * pred + t       (least squares over valid pixels)
    aligned_disp   =  max(s * pred + t,  1 / max_depth)
    aligned_depth  =  1 / aligned_disp

This is the protocol used for affine-invariant inverse-depth models in the
MiDaS line of work (Ranftl et al.). Table 2 of the Depth Anything V2 paper
does not spell out its alignment, so this remains a documented assumption
(see run_log.md).

History: the first version of this file fitted `gt_depth ~= s * pred + t`
(depth space). On a synthetic prediction that is *exactly* correct up to
scale and shift, that version scored AbsRel ~0.84 / delta1 ~0.13 and
dropped ~21% of pixels as non-positive. It was found in code review before
any real run; `fit_space="depth"` is kept only so the sensitivity of the
DIODE numbers to this choice can be reported, never as the primary result.
"""
from __future__ import annotations

import numpy as np

# DIODE's scanner range is quoted as up to ~350 m. Used only to floor the
# aligned disparity so that 1/disp cannot blow up. Documented assumption.
DEFAULT_MAX_DEPTH = 350.0
DEFAULT_MIN_DEPTH = 1e-3


def least_squares_scale_shift(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Solve min_{s,t} sum_{mask} (s*pred + t - target)^2 in closed form."""
    n_valid = int(mask.sum())
    if n_valid < 10:
        raise ValueError(f"Too few valid pixels to fit alignment: {n_valid}")

    p = pred[mask].astype(np.float64)
    g = target[mask].astype(np.float64)
    A = np.stack([p, np.ones_like(p)], axis=1)  # (N, 2)
    solution, *_ = np.linalg.lstsq(A, g, rcond=None)
    s, t = solution
    return float(s), float(t)


def align_disparity_to_depth(
    pred_disp: np.ndarray,
    gt_depth: np.ndarray,
    mask: np.ndarray,
    max_depth: float = DEFAULT_MAX_DEPTH,
) -> tuple[np.ndarray, float, float, int]:
    """Primary protocol: fit (s, t) in disparity space, then invert.

    Returns (aligned_depth, s, t, n_clipped) where n_clipped counts valid
    pixels whose aligned disparity had to be floored at 1/max_depth.
    """
    gt_disp = np.zeros_like(gt_depth, dtype=np.float64)
    gt_disp[mask] = 1.0 / gt_depth[mask]
    s, t = least_squares_scale_shift(pred_disp, gt_disp, mask)

    disp = s * pred_disp.astype(np.float64) + t
    floor = 1.0 / max_depth
    n_clipped = int(np.sum(mask & (disp < floor)))
    disp = np.maximum(disp, floor)
    return 1.0 / disp, s, t, n_clipped


def align_depth_space(
    pred_disp: np.ndarray, gt_depth: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Sensitivity-only protocol: fit depth = s * pred + t directly (the
    original, incorrect choice). Returned depth can be <= 0; callers must
    mask those pixels and report how many there were."""
    s, t = least_squares_scale_shift(pred_disp, gt_depth, mask)
    return s * pred_disp.astype(np.float64) + t, s, t


def align_prediction(pred_disp: np.ndarray, gt_depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Convenience wrapper: primary-protocol aligned depth only."""
    aligned, *_ = align_disparity_to_depth(pred_disp, gt_depth, mask)
    return aligned
