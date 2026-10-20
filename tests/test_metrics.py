"""
Offline unit tests for the student-written code: no checkpoints, no GPU,
no torch. Run with:  pytest tests/ -v
"""
import json

import numpy as np
import pytest

from src.alignment import align_depth_space, align_disparity_to_depth, align_prediction, least_squares_scale_shift
from src.da2k_dataset import infer_scene_type, load_da2k_annotations
from src.decision_rule import DecisionInputs, Verdict, apply_decision_rule
from src.metrics import (
    DA2KPairResult, abs_rel, bootstrap_ci, bootstrap_clustered_paired_accuracy_diff,
    bootstrap_paired_difference, da2k_accuracy, da2k_accuracy_by_scene, da2k_pair_correct, delta1,
    per_image_counts,
)

# ---------------------------------------------------------------------------
# alignment.py
# ---------------------------------------------------------------------------


def _scene(seed=0, near=1.0, far=80.0, shape=(120, 160)):
    rng = np.random.default_rng(seed)
    return 1.0 / rng.uniform(1 / far, 1 / near, size=shape)


def test_least_squares_recovers_known_transform():
    rng = np.random.default_rng(0)
    target = rng.uniform(1, 10, size=(50, 50))
    pred = (target - 0.3) / 2.5
    s, t = least_squares_scale_shift(pred, target, np.ones_like(target, bool))
    assert s == pytest.approx(2.5, rel=1e-6)
    assert t == pytest.approx(0.3, abs=1e-6)


def test_too_few_pixels_raises():
    mask = np.zeros((10, 10), bool)
    mask[0, :5] = True
    with pytest.raises(ValueError):
        least_squares_scale_shift(np.ones((10, 10)), np.ones((10, 10)), mask)


def test_perfect_affine_disparity_gives_perfect_metrics():
    """The regression test for the original bug: a prediction that is exactly
    right up to scale and shift IN DISPARITY must score perfectly."""
    gt = _scene()
    pred = 3.0 * (1.0 / gt) + 0.5
    mask = np.ones_like(gt, bool)
    aligned, s, t, n_clipped = align_disparity_to_depth(pred, gt, mask)
    assert abs_rel(aligned, gt, mask) == pytest.approx(0.0, abs=1e-9)
    assert delta1(aligned, gt, mask) == pytest.approx(1.0)
    assert n_clipped == 0


def test_depth_space_fit_is_wrong_on_perfect_disparity():
    """Documents why the depth-space fit is only a sensitivity number."""
    gt = _scene()
    pred = 3.0 * (1.0 / gt) + 0.5
    mask = np.ones_like(gt, bool)
    aligned, _, _ = align_depth_space(pred, gt, mask)
    positive = mask & (aligned > 1e-6)
    assert abs_rel(aligned, gt, positive) > 0.3


def test_alignment_respects_mask():
    gt = _scene(seed=1)
    pred = 2.0 * (1.0 / gt) + 0.1
    mask = np.ones_like(gt, bool)
    mask[:10] = False
    pred[:10] = 1e6  # garbage outside the mask must not affect the fit
    aligned, *_ = align_disparity_to_depth(pred, gt, mask)
    assert abs_rel(aligned, gt, mask) == pytest.approx(0.0, abs=1e-9)


def test_aligned_disparity_is_floored_and_counted():
    rng = np.random.default_rng(4)
    gt = rng.uniform(2.0, 10.0, size=(20, 20))
    pred = 1.0 / gt
    gt[0, :] = 90.0
    pred[0, :] = -0.05  # far row predicted "beyond infinity" -> negative aligned disparity
    mask = np.ones_like(gt, bool)
    aligned, _, _, n_clipped = align_disparity_to_depth(pred, gt, mask, max_depth=100.0)
    assert np.all(np.isfinite(aligned)) and np.all(aligned > 0) and aligned.max() <= 100.0 + 1e-9
    assert n_clipped >= 20


def test_align_prediction_wrapper():
    gt = _scene(seed=2)
    pred = 1.0 / gt
    assert np.allclose(align_prediction(pred, gt, np.ones_like(gt, bool)), gt)


# ---------------------------------------------------------------------------
# metrics.py: Track A
# ---------------------------------------------------------------------------


def test_abs_rel_zero_for_perfect_prediction():
    gt = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs_rel(gt.copy(), gt, np.ones(4, bool)) == pytest.approx(0.0)


def test_abs_rel_known_value():
    gt = np.array([2.0, 4.0])
    pred = np.array([1.0, 5.0])  # 0.5 and 0.25 -> 0.375
    assert abs_rel(pred, gt, np.ones(2, bool)) == pytest.approx(0.375)


def test_delta1_values():
    gt = np.array([10.0, 10.0])
    assert delta1(np.array([10.5, 9.5]), gt, np.ones(2, bool)) == pytest.approx(1.0)
    assert delta1(np.array([10.0, 20.0]), gt, np.ones(2, bool)) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# metrics.py: Track B
# ---------------------------------------------------------------------------


def _pred_map(v_point1, v_point2):
    pred = np.zeros((10, 10))
    pred[2, 2], pred[7, 7] = v_point1, v_point2
    return pred


def test_da2k_pair_scoring_conventions():
    assert da2k_pair_correct(_pred_map(5, 1), (2, 2), (7, 7), "point1", higher_is_closer=True) is True
    assert da2k_pair_correct(_pred_map(1, 5), (2, 2), (7, 7), "point1", higher_is_closer=True) is False
    assert da2k_pair_correct(_pred_map(1, 5), (2, 2), (7, 7), "point1", higher_is_closer=False) is True


def test_da2k_tie_counts_as_incorrect_for_both_labels():
    pred = _pred_map(3, 3)
    assert da2k_pair_correct(pred, (2, 2), (7, 7), "point1") is False
    assert da2k_pair_correct(pred, (2, 2), (7, 7), "point2") is False


def test_da2k_accuracy_and_by_scene():
    results = [
        DA2KPairResult("img1", "indoor", True, (0, 0), (1, 1), "point1"),
        DA2KPairResult("img1", "indoor", False, (0, 0), (1, 1), "point1"),
        DA2KPairResult("img2", "outdoor", True, (0, 0), (1, 1), "point1"),
    ]
    assert da2k_accuracy(results) == pytest.approx(2 / 3)
    assert da2k_accuracy_by_scene(results) == {"indoor": 0.5, "outdoor": 1.0}
    assert per_image_counts(results) == {"img1": (1, 2, "indoor"), "img2": (1, 1, "outdoor")}
    with pytest.raises(ValueError):
        da2k_accuracy([])


# ---------------------------------------------------------------------------
# metrics.py: bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_ci_constant_and_variance():
    point, lo, hi = bootstrap_ci(np.full(100, 0.7), n_boot=200, seed=1)
    assert point == lo == hi == pytest.approx(0.7)
    rng = np.random.default_rng(0)
    _, lo1, hi1 = bootstrap_ci(rng.normal(0, 0.01, 500), n_boot=300, seed=1)
    _, lo2, hi2 = bootstrap_ci(rng.normal(0, 1.0, 500), n_boot=300, seed=1)
    assert (hi1 - lo1) < (hi2 - lo2)


def test_paired_difference_uses_pairing():
    """With strongly correlated per-image values the paired CI must be much
    narrower than the spread of either model on its own."""
    rng = np.random.default_rng(0)
    base = rng.uniform(0.02, 0.3, size=300)
    a = base - 0.003 + rng.normal(0, 0.001, 300)
    point, lo, hi = bootstrap_paired_difference(a, base, n_boot=500, seed=1)
    assert point == pytest.approx(-0.003, abs=5e-4)
    assert hi - lo < 0.002
    assert hi < 0


def test_clustered_paired_accuracy_matches_pair_level_point_estimate():
    correct_v2 = np.array([2, 1, 1, 3])
    correct_v1 = np.array([1, 1, 0, 2])
    n_pairs = np.array([2, 1, 1, 3])
    point, lo, hi = bootstrap_clustered_paired_accuracy_diff(correct_v2, correct_v1, n_pairs, n_boot=500, seed=0)
    assert point == pytest.approx((7 - 4) / 7)
    assert lo <= point <= hi


def test_clustered_bootstrap_is_wider_than_naive_pair_bootstrap_when_pairs_are_correlated():
    """If all pairs in an image succeed or fail together, resampling pairs
    overstates the information; resampling images must give a wider CI."""
    rng = np.random.default_rng(3)
    n_img = 200
    n_pairs = rng.integers(1, 6, size=n_img)
    img_ok_v2 = rng.random(n_img) < 0.9
    img_ok_v1 = rng.random(n_img) < 0.8
    k2, k1 = n_pairs * img_ok_v2, n_pairs * img_ok_v1
    _, lo_c, hi_c = bootstrap_clustered_paired_accuracy_diff(k2, k1, n_pairs, n_boot=1000, seed=0)
    pair_a = np.repeat(img_ok_v2.astype(float), n_pairs)
    pair_b = np.repeat(img_ok_v1.astype(float), n_pairs)
    _, lo_p, hi_p = bootstrap_paired_difference(pair_a, pair_b, n_boot=1000, seed=0)
    assert (hi_c - lo_c) > (hi_p - lo_p)


# ---------------------------------------------------------------------------
# da2k_dataset.py
# ---------------------------------------------------------------------------


def test_scene_type_from_path():
    assert infer_scene_type("images/transparent_reflective/0001.jpg") == "transparent_reflective"
    assert infer_scene_type("images/misc/0001.jpg") == "unknown"


def test_unknown_scene_types_stop_the_loader(tmp_path):
    ann = {"images/indoor/a.jpg": [{"point1": [1, 2], "point2": [3, 4], "closer_point": "point1"}],
           "images/misc/b.jpg": [{"point1": [1, 2], "point2": [3, 4], "closer_point": "point1"}]}
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(ann))
    with pytest.raises(ValueError):
        load_da2k_annotations(path)
    assert len(load_da2k_annotations(path, allow_unknown_scenes=True)) == 2


# ---------------------------------------------------------------------------
# decision_rule.py  (thresholds and wording as pre-registered in the proposal)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gain, d_absrel, d_delta1, expected", [
    (6.8, -0.003, 0.003, Verdict.STRONG),          # paper reference values
    (5.0, 0.01, -0.02, Verdict.STRONG),            # thresholds are inclusive
    (6.0, -0.03, 0.005, Verdict.PARTIAL),          # one magnitude threshold missed
    (6.0, -0.03, 0.03, Verdict.PARTIAL),           # TWO missed, direction holds ("at least one")
    (3.0, 0.0, 0.0, Verdict.PARTIAL),              # gain below 5 pp, DIODE flat
    (1.0, -0.03, 0.05, Verdict.NOT_REPRODUCED),    # DIODE improves more than DA-2K
    (-2.0, 0.0, 0.0, Verdict.NOT_REPRODUCED),      # wrong direction on DA-2K
    (0.0, 0.0, 0.0, Verdict.NOT_REPRODUCED),       # no DA-2K gain at all
])
def test_decision_rule(gain, d_absrel, d_delta1, expected):
    verdict, reason = apply_decision_rule(DecisionInputs(gain, d_absrel, d_delta1))
    assert verdict == expected, reason
