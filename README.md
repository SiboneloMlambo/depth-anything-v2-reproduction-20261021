# Reproducing Depth Anything V2's benchmark-sensitivity claim

COMS7050A reproduction project. Sibonelo Mlambo.

## The claim being tested

Depth Anything V2 (Yang et al., NeurIPS 2024) argues that conventional depth
benchmarks can hide visible improvements. This repository tests one claim:
the V1 → V2 improvement is large on DA-2K ordinal accuracy (Table 3) but
small or mixed on DIODE AbsRel/delta1 (Table 2). The claim *is* that
contrast. Table 3 is the primary evidence and Table 2 the comparison baseline.

- **H0:** V2-S's DA-2K gain over V1-S is not meaningfully larger than its DIODE change.
- **H1:** V2-S has a large positive DA-2K gain while DIODE AbsRel/delta1 stay close or change inconsistently.

The pre-registered rule (`src/decision_rule.py`) uses the proposal's thresholds:
DA-2K gain ≥ 5 pp, |ΔAbsRel| ≤ 0.01, |Δdelta1| ≤ 0.02. The AbsRel and delta1
thresholds are absolute differences of fractions, so 0.02 means 2 points of delta1.

**Reference values, not results.** The paper's values are reference targets,
never reported as reproduced: DA-2K 88.5% (V1) → 95.3% (V2-S), and DIODE
AbsRel 0.076 → 0.073 and delta1 0.939 → 0.942, as transcribed from Tables 2–3.
The 88.5% V1 DA-2K figure is not clearly labelled with an encoder size and
may not be ViT-S. So this repository's **size-matched V1-S DA-2K accuracy,
and hence the size-matched V1-S → V2-S DA-2K gain, is a number the paper
does not report.**

## Numbers this repository adds that the paper does not have

- The V1-S DA-2K accuracy and the size-matched V1-S → V2-S DA-2K gain
- The DA-2K gain in each of the eight scenarios
- DIODE results split into indoor and outdoor
- Paired 95% bootstrap CIs over images for every V2 − V1 change (DA-2K resamples whole images)
- DIODE under the depth-space alignment, as a sensitivity check on the protocol

## Route: authors' models, student-written evaluation

- **From the authors**, cloned at pinned commits by `scripts/setup_external.sh`:
  - [`LiheYoung/Depth-Anything`](https://github.com/LiheYoung/Depth-Anything) @ `1d03336`
  - [`DepthAnything/Depth-Anything-V2`](https://github.com/DepthAnything/Depth-Anything-V2) @ `a561b84`

  These supply the model definitions, each release's own preprocessing, and the released ViT-S checkpoints.
- **Student-written:**
  - data loading (`src/da2k_dataset.py`, `src/diode_dataset.py`)
  - alignment (`src/alignment.py`)
  - metrics and bootstrap CIs (`src/metrics.py`)
  - both evaluation tracks
  - smoke test
  - decision rule
  - downloads and hash checks
  - figures and results table
  - all tests

## Reproduce everything (one command)

From a clean checkout:

```bash
conda env create -f environment.yml
conda activate da2-repro
bash reproduce.sh
```

`reproduce.sh` runs, in order:

1. Clones the official repos at pinned commits.
2. Runs the unit tests and the dry-run pipeline test.
3. Downloads both checkpoints, DA-2K and DIODE val, checked against `manifests/SHA256SUMS`.
4. Runs the automatic 20-image smoke test on both tracks: strict loading, prediction direction from DA-2K's own labels, and DIODE alignment sanity. It stops if anything looks wrong.
5. Runs Track A on DIODE val (all 771 frames; it stops if the count differs).
6. Runs Track B on DA-2K (every annotated pair, all eight scenarios).
7. Applies the decision rule, then writes `results/figures/*.png` and `results/results_table.md`.

Any stage with outputs already on disk is skipped, so after a Colab disconnect
you just run it again. `FORCE=1 bash reproduce.sh` re-runs both tracks. Free
Colab T4: open `notebooks/colab_reproduction.ipynb` and run its cells. It
uses Colab's preinstalled packages, records their versions in
`results/colab_environment.txt`, and calls `bash reproduce.sh`. The reported
results were produced this way, and `environment.yml` / `requirements.txt`
pin those same versions.

The SHA256 of each checkpoint and archive is committed in `manifests/SHA256SUMS`,
so every run verifies its downloads against the ones used for the reported results.

Quick check without a GPU, checkpoints or data:

```bash
pytest tests/ -v      # 35 tests, including an end-to-end dry run with fake models
```

## Outputs

| File | What it holds |
|---|---|
| `results/results_table.md` | Our numbers next to the paper's, the additional numbers, the sensitivity table and the verdict |
| `results/final_verdict.json` | Pre-registered verdict, its inputs, and the CIs (reported, not used by the rule) |
| `results/da2k/da2k_comparison.json` | Per-model accuracy overall and per scenario; V2 − V1 gains with cluster-bootstrap CIs |
| `results/diode/diode_comparison.json` | Per-model AbsRel/delta1 for indoor, outdoor and combined; paired V2 − V1 changes with CIs |
| `results/*/…per_pair*/per_image*.json` | Raw per-pair and per-image results behind every number |
| `results/smoke/smoke_report.json` | Smoke-test checks |
| `results/figures/fig1..fig3*.png` | Headline change, DA-2K by scenario, DIODE by split |

## Repository layout

```
reproduce.sh              the single documented command
environment.yml           conda environment (pins come from requirements.txt)
requirements.txt          package versions used for the reported results
scripts/setup_external.sh clones the official repos at pinned commits
manifests/                SHA256SUMS and download sources for the reported results
src/
  model_loader.py         the only use of the authors' code: strict per-release loading + inference
  alignment.py            disparity-space scale-and-shift alignment (depth-space kept for sensitivity)
  metrics.py              AbsRel, delta1, DA-2K pair scoring, paired and cluster bootstrap CIs
  da2k_dataset.py         DA-2K loader; fails on unknown scenes or missing images
  diode_dataset.py        DIODE val loader; indoor/outdoor split; 771-frame check
  run_track_a_diode.py    Track A
  run_track_b_da2k.py     Track B
  smoke_test.py           automatic 20-image checks on both tracks
  decision_rule.py        pre-registered rule (proposal wording quoted in the docstring)
  apply_decision.py       applies the rule to the full-run results
  make_figures.py         all figures and the results table
  download.py             downloads + hash manifest
tests/                    unit tests, manifest tests, end-to-end dry run with fake models
notebooks/                Colab runner (calls reproduce.sh)
data/README.md            licence and provenance notes
run_log.md                every guess, bug, failure and deviation
```

