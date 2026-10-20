"""
Metric implementations for both reproduction tracks (student-written).

Track A (DIODE, conventional): AbsRel and delta1 on aligned depth.
Track B (DA-2K, benchmark sensitivity): point-pair ordinal-depth accuracy.

Units: AbsRel, delta1 and DA-2K accuracy are all FRACTIONS in [0, 1].
Only the DA-2K V1->V2 gain is converted to percentage points, and only
where the decision rule asks for it.

Uncertainty: every V2-minus-V1 difference gets a *paired* bootstrap CI
over IMAGES (the independent unit). DA-2K images can carry more than one
annotated pair, so DA-2K resamples whole images (a cluster bootstrap),
never individual pairs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Track A: DIODE conventional metrics
# ---------------------------------------------------------------------------


def abs_rel(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """Mean absolute relative error: mean(|pred - gt| / gt) over valid pixels."""
    p, g = pred[mask], gt[mask]
    return float(np.mean(np.abs(p - g) / g))


def delta1(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of valid pixels with max(pred/gt, gt/pred) < 1.25."""
    p, g = pred[mask], gt[mask]
    ratio = np.maximum(p / g, g / p)
    return float(np.mean(ratio < 1.25))


@dataclass
class DiodeImageResult:
    image_id: str            # path relative to the DIODE val root (unique)
    split: str               # "indoors" or "outdoor"
    abs_rel: float           # primary protocol (disparity-space fit)
    delta1: float
    n_valid_px: int
    n_clipped_px: int        # valid pixels whose aligned disparity was floored
    scale: float
    shift: float
    # Sensitivity only: the depth-space fit (the original, wrong protocol).
    abs_rel_depthfit: float | None = None
    delta1_depthfit: float | None = None
    n_nonpositive_depthfit: int | None = None


# ---------------------------------------------------------------------------
# Track B: DA-2K ordinal-depth accuracy
# ---------------------------------------------------------------------------


@dataclass
class DA2KPairResult:
    image_id: str
    scene_type: str
    correct: bool
    point1: tuple[int, int]
    point2: tuple[int, int]
    closer_point: str
    tie: bool = False


def da2k_pair_prediction(
    pred_disp: np.ndarray,
    point1_hw: tuple[int, int],
    point2_hw: tuple[int, int],
    higher_is_closer: bool = True,
) -> str | None:
    """Return "point1", "point2", or None for an exact tie."""
    v1 = pred_disp[point1_hw[0], point1_hw[1]]
    v2 = pred_disp[point2_hw[0], point2_hw[1]]
    if v1 == v2:
        return None
    if higher_is_closer:
        return "point1" if v1 > v2 else "point2"
    return "point1" if v1 < v2 else "point2"


def da2k_pair_correct(
    pred_disp: np.ndarray,
    point1_hw: tuple[int, int],
    point2_hw: tuple[int, int],
    closer_point: str,
    higher_is_closer: bool = True,
) -> bool:
    """Score one DA-2K pair. An exact tie counts as INCORRECT (the model did
    not order the points); ties are also counted separately so they are visible.

    higher_is_closer=True matches the authors' disparity convention; the
    smoke test checks it automatically against DA-2K's own labels.
    """
    predicted = da2k_pair_prediction(pred_disp, point1_hw, point2_hw, higher_is_closer)
    return predicted is not None and predicted == closer_point


def da2k_accuracy(results: list[DA2KPairResult]) -> float:
    if not results:
        raise ValueError("No DA-2K results to score.")
    return float(np.mean([r.correct for r in results]))


def da2k_accuracy_by_scene(results: list[DA2KPairResult]) -> dict[str, float]:
    out = {}
    for scene in sorted(set(r.scene_type for r in results)):
        out[scene] = float(np.mean([r.correct for r in results if r.scene_type == scene]))
    return out


def per_image_counts(results: list[DA2KPairResult]) -> dict[str, tuple[int, int, str]]:
    """image_id -> (n_correct, n_pairs, scene_type)."""
    out: dict[str, list] = {}
    for r in results:
        entry = out.setdefault(r.image_id, [0, 0, r.scene_type])
        entry[0] += int(r.correct)
        entry[1] += 1
    return {k: tuple(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Uncertainty: bootstrap confidence intervals
# ---------------------------------------------------------------------------


def _percentile_ci(samples: np.ndarray, ci: float) -> tuple[float, float]:
    alpha = (1 - ci) / 2
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1 - alpha))


def bootstrap_ci(values, n_boot: int = 2000, ci: float = 0.95, seed: int = 0, statistic=np.mean):
    """Percentile bootstrap CI for one model's per-image statistic.
    Returns (point_estimate, lower, upper)."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        raise ValueError("Cannot bootstrap an empty array.")
    rng = np.random.default_rng(seed)
    n = len(values)
    point = float(statistic(values))
    boot = np.array([statistic(values[rng.integers(0, n, size=n)]) for _ in range(n_boot)])
    lo, hi = _percentile_ci(boot, ci)
    return point, lo, hi


def bootstrap_paired_difference(values_a, values_b, n_boot: int = 2000, ci: float = 0.95, seed: int = 0):
    """Paired bootstrap CI for mean(a) - mean(b), where a[i] and b[i] are the
    SAME image under two models (used for DIODE per-image AbsRel/delta1)."""
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    if values_a.shape != values_b.shape:
        raise ValueError("Paired arrays must have the same shape.")
    if len(values_a) == 0:
        raise ValueError("Cannot bootstrap an empty array.")
    rng = np.random.default_rng(seed)
    n = len(values_a)
    diff = values_a - values_b
    point = float(diff.mean())
    boot = np.array([diff[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)])
    lo, hi = _percentile_ci(boot, ci)
    return point, lo, hi


def bootstrap_clustered_paired_accuracy_diff(
    correct_a, correct_b, n_pairs, n_boot: int = 2000, ci: float = 0.95, seed: int = 0
):
    """Paired cluster bootstrap for a pair-level accuracy difference.

    correct_a[i], correct_b[i]: number of correctly ordered pairs in image i
    under model A and model B; n_pairs[i]: annotated pairs in image i.
    The statistic is sum(correct_a - correct_b) / sum(n_pairs), i.e. exactly
    the pair-level accuracy difference, while resampling whole images.
    Returns (point, lower, upper) as fractions.
    """
    a = np.asarray(correct_a, dtype=np.float64)
    b = np.asarray(correct_b, dtype=np.float64)
    n = np.asarray(n_pairs, dtype=np.float64)
    if not (a.shape == b.shape == n.shape):
        raise ValueError("Paired arrays must have the same shape.")
    if len(a) == 0:
        raise ValueError("Cannot bootstrap an empty array.")
    rng = np.random.default_rng(seed)
    k = len(a)
    d = a - b
    point = float(d.sum() / n.sum())
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, k, size=k)
        boot[i] = d[idx].sum() / n[idx].sum()
    lo, hi = _percentile_ci(boot, ci)
    return point, lo, hi


def bootstrap_clustered_accuracy(correct, n_pairs, n_boot: int = 2000, ci: float = 0.95, seed: int = 0):
    """Cluster (image-level) bootstrap CI for one model's pair-level accuracy."""
    zeros = np.zeros(len(correct))
    return bootstrap_clustered_paired_accuracy_diff(correct, zeros, n_pairs, n_boot, ci, seed)
