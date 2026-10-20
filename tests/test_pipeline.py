"""
End-to-end dry run of the whole pipeline (smoke test -> Track A -> Track B ->
decision rule -> figures) on tiny synthetic DIODE and DA-2K folders, with the
authors' models replaced by fakes. This checks the plumbing and the
student-written code paths without torch, checkpoints or real data.

The fakes read a disparity map that is encoded in the image's red channel:
"V2" returns it exactly, "V1" adds noise. So V2 should beat V1 on both
benchmarks, the smoke test should pass, and a sign flip must be caught.
"""
import json
import zlib

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("matplotlib")

from src import apply_decision, make_figures, run_track_a_diode, run_track_b_da2k, smoke_test  # noqa: E402
from src.da2k_dataset import SCENE_TYPES  # noqa: E402


class FakeModel:
    def __init__(self, version):
        self._reproduction_version = version


def fake_infer(model, raw_image_bgr, input_size=518, sign=1.0):
    disp = raw_image_bgr[..., 2].astype(np.float64) / 255.0
    if model._reproduction_version == "v1":
        seed = zlib.crc32(raw_image_bgr.tobytes()) & 0xFFFF
        disp = disp + np.random.default_rng(seed).normal(0, 0.02, disp.shape)
    return sign * disp


def _write_image(path, disp):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros(disp.shape + (3,), np.uint8)
    img[..., 2] = np.clip(np.round(disp * 255), 1, 255).astype(np.uint8)
    cv2.imwrite(str(path), img)


def make_diode(root, n_per_split=4, shape=(48, 64)):
    rng = np.random.default_rng(0)
    for split in ("indoors", "outdoor"):
        for k in range(n_per_split):
            d = root / "val" / split / f"scene_{k:05d}" / "scan_00001"
            stem = f"{k:05d}_00001_{split}_000_000"
            yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
            depth = 1.5 + 15.0 * (yy / shape[0]) + rng.uniform(0, 0.5) * (xx / shape[1])
            _write_image(d / f"{stem}.png", 1.0 / depth)
            np.save(d / f"{stem}_depth.npy", depth[..., None].astype(np.float32))
            mask = np.ones(shape, np.float32)
            mask[:2] = 0
            np.save(d / f"{stem}_depth_mask.npy", mask)
    return root / "val"


def make_da2k(root, images_per_scene=3, shape=(48, 64)):
    rng = np.random.default_rng(1)
    annotations = {}
    for scene in SCENE_TYPES:
        for k in range(images_per_scene):
            rel = f"images/{scene}/{k:03d}.png"
            yy, _ = np.mgrid[0:shape[0], 0:shape[1]]
            disp = 0.1 + 0.85 * (yy / shape[0])   # lower rows are closer
            _write_image(root / rel, disp)
            pairs = []
            for _ in range(int(rng.integers(1, 4))):
                near = [int(rng.integers(30, 47)), int(rng.integers(0, 63))]
                far = [int(rng.integers(0, 15)), int(rng.integers(0, 63))]
                pairs.append({"point1": near, "point2": far, "closer_point": "point1"})
            annotations[rel] = pairs
    (root / "annotations.json").write_text(json.dumps(annotations))
    return root


@pytest.fixture
def fake_models(monkeypatch):
    def install(sign=1.0):
        loader = lambda ckpt, model_version, device=None: FakeModel(model_version)  # noqa: E731
        infer = lambda m, img, input_size=518: fake_infer(m, img, input_size, sign)  # noqa: E731
        for module in (run_track_a_diode, run_track_b_da2k, smoke_test):
            monkeypatch.setattr(module, "load_depth_anything_vits", loader)
            monkeypatch.setattr(module, "infer_disparity", infer)
    return install


def test_full_pipeline_dry_run(tmp_path, fake_models):
    fake_models()
    diode_root = make_diode(tmp_path / "diode")
    da2k_root = make_da2k(tmp_path / "da2k")
    results = tmp_path / "results"
    ck = ["--checkpoint-v1", "v1.pth", "--checkpoint-v2", "v2.pth"]

    smoke_test.main(["--diode-root", str(diode_root), "--da2k-root", str(da2k_root), *ck,
                     "--outdir", str(results / "smoke")])
    assert json.loads((results / "smoke" / "smoke_report.json").read_text())["passed"]

    run_track_a_diode.main(["--diode-root", str(diode_root), *ck, "--outdir", str(results / "diode"),
                            "--expected-frames", "8"])
    run_track_b_da2k.main(["--da2k-root", str(da2k_root), *ck, "--outdir", str(results / "da2k")])

    diode = json.loads((results / "diode" / "diode_comparison.json").read_text())
    da2k = json.loads((results / "da2k" / "da2k_comparison.json").read_text())
    assert set(diode["v2_minus_v1"]) >= {"indoors", "outdoor", "combined"}
    assert diode["v2s"]["combined"]["abs_rel_mean"] < 0.06          # exact up to 8-bit quantisation
    assert diode["delta_abs_rel"] < 0                                 # V2 better than noisy V1
    assert da2k["v2s_accuracy"] == pytest.approx(1.0)
    assert set(SCENE_TYPES) <= set(da2k["v2_minus_v1"])
    assert da2k["delta_da2k_pp"] == pytest.approx(da2k["v2_minus_v1"]["overall"]["gain_pp"])

    apply_decision.main(["--results-dir", str(results)])
    make_figures.main(["--results-dir", str(results)])
    verdict = json.loads((results / "final_verdict.json").read_text())
    assert verdict["verdict"] in {"strong reproduction", "partial reproduction", "not reproduced"}
    for name in ("fig1_v2_minus_v1.png", "fig2_da2k_by_scene.png", "fig3_diode_by_split.png"):
        assert (results / "figures" / name).stat().st_size > 1000
    assert "Numbers the paper does not report" in (results / "results_table.md").read_text()

    # Re-running a finished stage is skipped, not recomputed.
    stamp = (results / "da2k" / "da2k_comparison.json").stat().st_mtime_ns
    run_track_b_da2k.main(["--da2k-root", str(da2k_root), *ck, "--outdir", str(results / "da2k")])
    assert (results / "da2k" / "da2k_comparison.json").stat().st_mtime_ns == stamp


def test_smoke_test_catches_sign_flip(tmp_path, fake_models):
    fake_models(sign=-1.0)
    diode_root = make_diode(tmp_path / "diode")
    da2k_root = make_da2k(tmp_path / "da2k")
    with pytest.raises(SystemExit):
        smoke_test.main(["--diode-root", str(diode_root), "--da2k-root", str(da2k_root),
                         "--checkpoint-v1", "a", "--checkpoint-v2", "b", "--outdir", str(tmp_path / "smoke")])
    report = json.loads((tmp_path / "smoke" / "smoke_report.json").read_text())
    assert not report["passed"]
    assert any("sign flip" in f for f in report["failures"])


def test_decision_rule_refuses_smoke_results(tmp_path, fake_models):
    fake_models()
    diode_root = make_diode(tmp_path / "diode")
    da2k_root = make_da2k(tmp_path / "da2k")
    ck = ["--checkpoint-v1", "a", "--checkpoint-v2", "b"]
    run_track_a_diode.main(["--diode-root", str(diode_root), *ck, "--smoke-test", "--outdir", str(tmp_path / "r" / "diode")])
    run_track_b_da2k.main(["--da2k-root", str(da2k_root), *ck, "--smoke-test", "--outdir", str(tmp_path / "r" / "da2k")])
    with pytest.raises(SystemExit):
        apply_decision.main(["--results-dir", str(tmp_path / "r")])


def test_track_a_stops_on_wrong_frame_count(tmp_path, fake_models):
    fake_models()
    diode_root = make_diode(tmp_path / "diode")
    with pytest.raises(SystemExit):
        run_track_a_diode.main(["--diode-root", str(diode_root), "--checkpoint-v1", "a", "--checkpoint-v2", "b",
                                "--outdir", str(tmp_path / "out")])   # default expects 771
