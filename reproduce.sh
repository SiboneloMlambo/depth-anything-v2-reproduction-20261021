#!/usr/bin/env bash
# One command from a clean checkout to every number and figure in the write-up.
#
#   conda env create -f environment.yml && conda activate da2-repro
#   bash reproduce.sh
#
# Stages whose outputs already exist are skipped, so an interrupted run can
# simply be restarted. FORCE=1 bash reproduce.sh re-runs both evaluation tracks.

#set -euo pipefail, This makes the script fail fast instead of continuing in a broken state:
#-e exits as soon as any command fails.
#-u treats an undefined variable as an error, so typos dont silently become empty strings.
#-o pipefail makes a pipeline fail if any command in it fails, not only the last one.

set -euo pipefail
cd "$(dirname "$0")"

CKPT_V1=checkpoints/depth_anything_v1_vits.pth
CKPT_V2=checkpoints/depth_anything_v2_vits.pth
FORCE_FLAG=""
if [ "${FORCE:-0}" = "1" ]; then FORCE_FLAG="--force"; fi

echo "== 1/7 Official model code at pinned commits"
bash scripts/setup_external.sh

echo "== 2/7 Unit tests (student-written metrics, alignment, decision rule)"
python -m pytest -q tests/

echo "== 3/7 Checkpoints and data, verified against manifests/SHA256SUMS"
python -m src.download

echo "== 4/7 Smoke test: strict loading, prediction direction, alignment sanity (20 images per track)"
python -m src.smoke_test --diode-root data/diode/val --da2k-root data/da2k \
  --checkpoint-v1 "$CKPT_V1" --checkpoint-v2 "$CKPT_V2"

echo "== 5/7 Track A: DIODE val (771 frames)"
python -m src.run_track_a_diode --diode-root data/diode/val \
  --checkpoint-v1 "$CKPT_V1" --checkpoint-v2 "$CKPT_V2" $FORCE_FLAG

echo "== 6/7 Track B: DA-2K (all images, all eight scenarios)"
python -m src.run_track_b_da2k --da2k-root data/da2k \
  --checkpoint-v1 "$CKPT_V1" --checkpoint-v2 "$CKPT_V2" $FORCE_FLAG

echo "== 7/7 Pre-registered decision rule, figures and results table"
python -m src.apply_decision --results-dir results
python -m src.make_figures --results-dir results

echo "Done. See results/results_table.md, results/final_verdict.json and results/figures/."
