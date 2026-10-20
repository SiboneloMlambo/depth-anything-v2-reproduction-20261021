"""
Pre-registered decision rule.

Proposal wording (September 2026), verbatim:

    "A strong reproduction requires DA-2K gain >= 5 points, |change in
    AbsRel| <= 0.01 and |change in delta1| <= 0.02. A partial reproduction
    has the predicted direction but misses at least one magnitude
    threshold; it is not reproduced if no meaningful DA-2K-versus-DIODE
    asymmetry appears."

Units: the DA-2K gain is in percentage points; the AbsRel and delta1
changes are ABSOLUTE differences of fractions (0.02 means 2 percentage
points of delta1, not 2%).

Operational reading of "predicted direction" (written down 2026-10-20,
BEFORE any real result existed, and logged in run_log.md):

    V2 improves on DA-2K (gain > 0 pp), AND that improvement, expressed as
    a fraction of pairs, is larger than V2's improvement in DIODE delta1,
    expressed as a fraction of pixels. Both are proportions in [0, 1], so
    this is the one like-for-like comparison of the two benchmarks.

Every STRONG result satisfies this automatically (>= 0.05 vs <= 0.02).

History: the first version of this file used "exactly one threshold
missed" for PARTIAL, which did not match the proposal ("at least one").
Fixed in code review before any real run.

Do not edit the thresholds or this reading after seeing results. If they
turn out to be a poor fit to the data, say so in run_log.md instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Verdict(Enum):
    STRONG = "strong reproduction"
    PARTIAL = "partial reproduction"
    NOT_REPRODUCED = "not reproduced"


@dataclass
class DecisionInputs:
    delta_da2k_pp: float     # Accuracy(V2-S) - Accuracy(V1-S), percentage points
    delta_abs_rel: float     # AbsRel(V2-S) - AbsRel(V1-S), absolute (fraction)
    delta_delta1: float      # delta1(V2-S) - delta1(V1-S), absolute (fraction)

    da2k_threshold_pp: float = 5.0
    abs_rel_threshold: float = 0.01
    delta1_threshold: float = 0.02


def apply_decision_rule(inputs: DecisionInputs) -> tuple[Verdict, str]:
    da2k_pass = inputs.delta_da2k_pp >= inputs.da2k_threshold_pp
    abs_rel_pass = abs(inputs.delta_abs_rel) <= inputs.abs_rel_threshold
    delta1_pass = abs(inputs.delta_delta1) <= inputs.delta1_threshold

    da2k_gain_frac = inputs.delta_da2k_pp / 100.0
    direction_ok = inputs.delta_da2k_pp > 0 and da2k_gain_frac > inputs.delta_delta1

    checks = (
        f"da2k_pass={da2k_pass} (gain {inputs.delta_da2k_pp:+.2f} pp vs >= {inputs.da2k_threshold_pp}), "
        f"abs_rel_pass={abs_rel_pass} (|dAbsRel|={abs(inputs.delta_abs_rel):.4f} vs <= {inputs.abs_rel_threshold}), "
        f"delta1_pass={delta1_pass} (|d-delta1|={abs(inputs.delta_delta1):.4f} vs <= {inputs.delta1_threshold}), "
        f"direction_ok={direction_ok} (DA-2K gain {da2k_gain_frac:+.4f} vs delta1 change {inputs.delta_delta1:+.4f})"
    )

    if da2k_pass and abs_rel_pass and delta1_pass:
        return Verdict.STRONG, "All three magnitude thresholds met. " + checks
    if direction_ok:
        return Verdict.PARTIAL, "Predicted direction present but at least one magnitude threshold missed. " + checks
    return Verdict.NOT_REPRODUCED, "No meaningful DA-2K-versus-DIODE asymmetry in the predicted direction. " + checks


if __name__ == "__main__":
    # Self-check on the paper's own reference values (Tables 2 and 3, as
    # transcribed in the README). NOT a reproduced result. Note the
    # caveat in the README: the 88.5 V1 DA-2K figure may not be ViT-S.
    example = DecisionInputs(delta_da2k_pp=95.3 - 88.5, delta_abs_rel=0.073 - 0.076, delta_delta1=0.942 - 0.939)
    verdict, reason = apply_decision_rule(example)
    print(f"[Reference-values self-check, NOT a reproduced result] {verdict.value}")
    print(reason)
