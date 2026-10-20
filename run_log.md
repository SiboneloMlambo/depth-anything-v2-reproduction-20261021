# Run log

Record every guessed preprocessing detail, failed run and deviation from the
paper here as it happens, not at the end. The write-up's "what did not
reproduce" section is built from this file.

## 2026-10-20: code review before any real run

Found by review and synthetic tests. No real result existed at this point.

1. **DIODE alignment was fitted in the wrong space.** `src/alignment.py` fitted
   `depth ≈ s·disparity + t`. The models output affine-invariant *disparity*,
   and depth is not affine in disparity. On a synthetic image whose prediction
   is exactly right up to scale and shift, the old code scored AbsRel 0.842,
   delta1 0.126, and dropped 21% of pixels as non-positive. The new code fits
   `1/depth ≈ s·disparity + t`, floors the aligned disparity at 1/350 m, then
   inverts. The same synthetic image now scores 0.000 / 1.000. The depth-space
   fit is still computed, but only as a sensitivity number. Regression test:
   `test_perfect_affine_disparity_gives_perfect_metrics`.
2. **V1 could not be built from this repo's root.** The official V1 code loads
   DINOv2 with `torch.hub.load('torchhub/facebookresearch_dinov2_main', ...,
   source='local')`, a path relative to the current directory. The model is now
   built from inside `external/Depth-Anything` (`src/model_loader.py`).
3. **Decision-rule code did not match the proposal.** The proposal says a partial
   reproduction "misses **at least one** magnitude threshold". The code said
   "exactly one". Fixed to match the proposal.
4. **Operational reading of "predicted direction".** The proposal never defined
   this phrase, so it is fixed here, before any results: the DA-2K gain is
   positive, AND, as a fraction of pairs, it exceeds the DIODE delta1
   improvement as a fraction of pixels. Both are proportions, so this is the
   one like-for-like comparison. It is recorded in `src/decision_rule.py` and
   must not change after results are seen.
5. **Uncertainty did not match the proposal.** There were per-model CIs over
   pairs and no CI on the V2 − V1 change. Now every change has a paired
   bootstrap CI over images, and DA-2K resamples whole images.
6. **Also added:**
   - DIODE indoor/outdoor split
   - per-scenario DA-2K gains
   - a check that the DIODE val split has 771 frames
   - stopping on unknown DA-2K scenes or missing images
   - DA-2K ties counted as incorrect (and reported)
   - an automatic smoke test in place of hand-picked calibration points

## Checkpoint provenance

Filled automatically on the first run. `manifests/SHA256SUMS` holds the hashes;
`manifests/sources.json` holds the repo, revision and date.

| Checkpoint | Source | Notes |
|---|---|---|
| `checkpoints/depth_anything_v1_vits.pth` | HF Space `LiheYoung/Depth-Anything`, `checkpoints/depth_anything_vits14.pth` | Confirm that strict loading succeeds |
| `checkpoints/depth_anything_v2_vits.pth` | HF model `depth-anything/Depth-Anything-V2-Small`, `depth_anything_v2_vits.pth` | |

Official code pinned in `scripts/setup_external.sh`: V1 `1d03336771fe…`, V2 `a561b849ebae…`.

## Data licence and provenance (DA-2K)

- Date checked: 2026-09-14
- Licence found: Apache-2.0, per the dataset card at
  https://huggingface.co/datasets/depth-anything/DA-2K. It permits use and
  modification and requires attribution: cite "Depth Anything V2" (Yang et al.,
  arXiv:2406.09414, 2024).
- Access method: the images are hosted directly on the Hugging Face dataset repo
  (not links to originals): 1,035 images, about 682 MB, plus `annotations.json`.
- Decision: proceed.
- NOTE: these figures came from an automated page fetch, not a manual read of
  the dataset card. Confirm them on the page yourself before treating them as
  final, then delete this note. Also note what the card says about the
  provenance of the original images.

## Preprocessing decisions not stated in the paper

- **DIODE alignment:** per-image least squares in disparity space, then inverted
  (MiDaS-style protocol for affine-invariant inverse depth). Table 2 does not
  state its alignment, so this is an assumption.
- **Disparity floor:** 1/350 m, from DIODE's quoted scanner range. Assumption.
  The fraction of clipped pixels is reported per model in `diode_comparison.json`.
- **DIODE valid pixels:** official mask AND depth > 1 mm.
- **Input size:** 518 px for both models, each with its own release's
  preprocessing. V2 upsamples with `align_corners=True`, V1 with `False`. Both
  match the official code; this is a small pipeline difference between the two
  models, not something chosen here.
- **DA-2K ties:** exact ties count as incorrect. The count is reported.
- **Averaging:** DIODE metrics are averaged per image, then over images.
  The paper may average differently; unknown.

## Smoke test (`results/smoke/smoke_report.json`)

- Date:
- Passed? If not, which check failed:
- V1-S / V2-S smoke DA-2K accuracy (sign check):
- V1-S / V2-S smoke DIODE AbsRel, clipped fraction:

## Full run

- Date, Colab GPU, runtime:
- Interruptions / restarts:
- Package versions actually used (`pip freeze | grep -E "torch|numpy|opencv|huggingface"`); update `requirements.txt` if they differ:
- Final verdict (`results/final_verdict.json`):

## What didn't reproduce (fill in last; this should be most of the write-up)
