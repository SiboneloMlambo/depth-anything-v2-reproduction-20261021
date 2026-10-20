"""
Combines both tracks' outputs and applies the pre-registered decision rule.

Reads results/diode/diode_comparison.json and results/da2k/da2k_comparison.json.
Writes results/final_verdict.json.

Usage:
    python -m src.apply_decision --results-dir results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.decision_rule import DecisionInputs, apply_decision_rule


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    diode = json.loads((results_dir / "diode" / "diode_comparison.json").read_text())
    da2k = json.loads((results_dir / "da2k" / "da2k_comparison.json").read_text())
    if diode["protocol"]["smoke_test"] or da2k["protocol"]["smoke_test"]:
        raise SystemExit("Refusing to apply the decision rule to smoke-test results.")

    inputs = DecisionInputs(
        delta_da2k_pp=da2k["delta_da2k_pp"],
        delta_abs_rel=diode["delta_abs_rel"],
        delta_delta1=diode["delta_delta1"],
    )
    verdict, reason = apply_decision_rule(inputs)
    out = {
        "verdict": verdict.value,
        "reason": reason,
        "inputs": inputs.__dict__,
        "uncertainty_not_used_by_rule": {
            "da2k_gain_pp_ci95": da2k["v2_minus_v1"]["overall"]["gain_pp_ci95"],
            "diode_delta_abs_rel_ci95": diode["v2_minus_v1"]["combined"]["delta_abs_rel_ci95"],
            "diode_delta_delta1_ci95": diode["v2_minus_v1"]["combined"]["delta_delta1_ci95"],
        },
    }
    print(json.dumps(out, indent=2))
    (results_dir / "final_verdict.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
