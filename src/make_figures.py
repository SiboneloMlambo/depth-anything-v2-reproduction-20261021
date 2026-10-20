"""
Builds every figure and the results table used in the write-up from the
JSON outputs of both tracks. No numbers are typed in here except the
paper's reference values, which are labelled as such.

Outputs:
    results/figures/fig1_v2_minus_v1.png      headline: V2-V1 change on each benchmark
    results/figures/fig2_da2k_by_scene.png    DA-2K gain per scenario
    results/figures/fig3_diode_by_split.png   DIODE AbsRel / delta1 per split, per model
    results/results_table.md                  our numbers next to the paper's

Usage:
    python -m src.make_figures --results-dir results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.da2k_dataset import SCENE_TYPES  # noqa: E402

# Reference values as transcribed from the paper (Table 3: DA-2K; Table 2:
# DIODE). Verify against the PDF before submission. The V1 DA-2K figure
# may not be the ViT-S model -- see README.
PAPER_REFERENCE = {
    "da2k_v1": 0.885, "da2k_v2s": 0.953,
    "diode_abs_rel_v1s": 0.076, "diode_abs_rel_v2s": 0.073,
    "diode_delta1_v1s": 0.939, "diode_delta1_v2s": 0.942,
}

V1_COLOR, V2_COLOR, CHANGE_COLOR = "#2a78d6", "#eb6834", "#52514e"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _err(point, ci):
    return [[point - ci[0]], [ci[1] - point]]


def _xlim_with_room(ax, lows, highs, label_room=0.35):
    """Fit all CIs and zero, plus room on the right for value labels."""
    lo, hi = min(0.0, *lows), max(0.0, *highs)
    span = max(hi - lo, 1.0)
    ax.set_xlim(lo - 0.05 * span, hi + label_room * span)
    return 0.02 * span


def fig_headline(da2k, diode, out: Path):
    rows = [("DA-2K ordinal accuracy (all)", da2k["v2_minus_v1"]["overall"]["gain_pp"],
             da2k["v2_minus_v1"]["overall"]["gain_pp_ci95"])]
    for split, label in (("combined", "all"), ("indoors", "indoor"), ("outdoor", "outdoor")):
        d = diode["v2_minus_v1"].get(split)
        if d:
            rows.append((f"DIODE delta1 ({label})", 100 * d["delta_delta1"], [100 * v for v in d["delta_delta1_ci95"]]))
    fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(rows) + 1.4))
    pad = _xlim_with_room(ax, [r[2][0] for r in rows], [max(r[2][1], 5.0) for r in rows], 0.2)
    for i, (label, point, ci) in enumerate(rows):
        ax.barh(i, point, height=0.55, color=CHANGE_COLOR)
        ax.errorbar(point, i, xerr=_err(point, ci), fmt="none", ecolor=INK, capsize=3, linewidth=1)
        ax.text(max(point, ci[1]) + pad, i, f"{point:+.1f} pp", va="center", color=INK, fontsize=9)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.axvline(5, color=MUTED, linewidth=1, linestyle="--")
    # Label sits inside the axes at the bottom so it never collides with the title.
    ax.text(5, 0.02, " 5 pp DA-2K threshold", color=MUTED, fontsize=8, va="bottom", ha="left",
            transform=ax.get_xaxis_transform())
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("V2-S minus V1-S, percentage points (paired 95% CI over images)")
    ax.set_title("Change from V1-S to V2-S on each benchmark", loc="left", color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_da2k_by_scene(da2k, out: Path):
    scenes = [s for s in SCENE_TYPES if s in da2k["v2_minus_v1"]]
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(scenes) + 1.4))
    gains = [da2k["v2_minus_v1"][s] for s in scenes]
    pad = _xlim_with_room(ax, [g["gain_pp_ci95"][0] for g in gains], [g["gain_pp_ci95"][1] for g in gains], 0.45)
    for i, s in enumerate(scenes):
        d = da2k["v2_minus_v1"][s]
        ax.barh(i, d["gain_pp"], height=0.55, color=CHANGE_COLOR)
        ax.errorbar(d["gain_pp"], i, xerr=_err(d["gain_pp"], d["gain_pp_ci95"]), fmt="none",
                    ecolor=INK, capsize=3, linewidth=1)
        ax.text(max(d["gain_pp"], d["gain_pp_ci95"][1]) + pad, i, f"{d['gain_pp']:+.1f}  (n={d['n_pairs']} pairs)",
                va="center", color=INK, fontsize=8)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(range(len(scenes)), [s.replace("_", " ") for s in scenes])
    ax.invert_yaxis()
    ax.set_xlabel("DA-2K accuracy gain, V2-S minus V1-S (pp, cluster-bootstrap 95% CI)")
    ax.set_title("DA-2K gain by scenario", loc="left", color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_diode_by_split(diode, out: Path):
    splits = [s for s in ("combined", "indoors", "outdoor") if s in diode["v1s"]]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, metric, title in ((axes[0], "abs_rel", "AbsRel (lower is better)"),
                              (axes[1], "delta1", "delta1 (higher is better)")):
        for j, (name, label, color) in enumerate((("v1s", "V1-S", V1_COLOR), ("v2s", "V2-S", V2_COLOR))):
            ys = [i + (j - 0.5) * 0.36 for i in range(len(splits))]
            vals = [diode[name][s][f"{metric}_mean"] for s in splits]
            cis = [diode[name][s][f"{metric}_ci95"] for s in splits]
            ax.barh(ys, vals, height=0.34, color=color, label=label)
            for y, v, ci in zip(ys, vals, cis):
                ax.errorbar(v, y, xerr=_err(v, ci), fmt="none", ecolor=INK, capsize=2, linewidth=0.8)
        ax.set_yticks(range(len(splits)), ["all" if s == "combined" else s.replace("indoors", "indoor") for s in splits])
        ax.invert_yaxis()
        ax.set_title(title, loc="left", color=INK, fontsize=10)
        _style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper right", ncol=2)
    fig.suptitle("DIODE val, per-image scale-and-shift in disparity space (95% CI)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def results_table(da2k, diode, verdict, out: Path):
    P = PAPER_REFERENCE
    dc = diode["v2_minus_v1"]["combined"]
    g = da2k["v2_minus_v1"]["overall"]
    lines = [
        "# Results",
        "",
        "Generated by `src/make_figures.py` from `results/*/`. Paper values are reference targets as",
        "transcribed from Tables 2-3, not reproduced results. The paper's V1 DA-2K figure may not be ViT-S.",
        "",
        "| Quantity | Ours: V1-S | Ours: V2-S | Ours: V2-S - V1-S (95% CI) | Paper: V1 | Paper: V2-S |",
        "|---|---|---|---|---|---|",
        f"| DA-2K accuracy | {da2k['v1s_accuracy']:.3f} | {da2k['v2s_accuracy']:.3f} | "
        f"{g['gain_pp']:+.1f} pp [{g['gain_pp_ci95'][0]:+.1f}, {g['gain_pp_ci95'][1]:+.1f}] | "
        f"{P['da2k_v1']:.3f} | {P['da2k_v2s']:.3f} |",
        f"| DIODE AbsRel (all) | {diode['v1s']['combined']['abs_rel_mean']:.4f} | "
        f"{diode['v2s']['combined']['abs_rel_mean']:.4f} | {dc['delta_abs_rel']:+.4f} "
        f"[{dc['delta_abs_rel_ci95'][0]:+.4f}, {dc['delta_abs_rel_ci95'][1]:+.4f}] | "
        f"{P['diode_abs_rel_v1s']:.3f} | {P['diode_abs_rel_v2s']:.3f} |",
        f"| DIODE delta1 (all) | {diode['v1s']['combined']['delta1_mean']:.4f} | "
        f"{diode['v2s']['combined']['delta1_mean']:.4f} | {dc['delta_delta1']:+.4f} "
        f"[{dc['delta_delta1_ci95'][0]:+.4f}, {dc['delta_delta1_ci95'][1]:+.4f}] | "
        f"{P['diode_delta1_v1s']:.3f} | {P['diode_delta1_v2s']:.3f} |",
        "",
        "## Numbers the paper does not report",
        "",
        "| Quantity | V1-S | V2-S | V2-S - V1-S (95% CI) |",
        "|---|---|---|---|",
    ]
    for split in ("indoors", "outdoor"):
        if split in diode["v2_minus_v1"]:
            d = diode["v2_minus_v1"][split]
            for metric in ("abs_rel", "delta1"):
                lines.append(
                    f"| DIODE {metric} ({split}) | {diode['v1s'][split][f'{metric}_mean']:.4f} | "
                    f"{diode['v2s'][split][f'{metric}_mean']:.4f} | {d[f'delta_{metric}']:+.4f} "
                    f"[{d[f'delta_{metric}_ci95'][0]:+.4f}, {d[f'delta_{metric}_ci95'][1]:+.4f}] |")
    for s in SCENE_TYPES:
        if s in da2k["v2_minus_v1"]:
            d = da2k["v2_minus_v1"][s]
            lines.append(f"| DA-2K {s} (n={d['n_pairs']} pairs) | {da2k['v1s'][s]['accuracy']:.3f} | "
                         f"{da2k['v2s'][s]['accuracy']:.3f} | {d['gain_pp']:+.1f} pp "
                         f"[{d['gain_pp_ci95'][0]:+.1f}, {d['gain_pp_ci95'][1]:+.1f}] |")
    sens_v1 = diode["v1s"]["combined"]["sensitivity_depth_space_fit"]
    sens_v2 = diode["v2s"]["combined"]["sensitivity_depth_space_fit"]
    lines += [
        "",
        "## Sensitivity: DIODE with the depth-space alignment (wrong protocol, for comparison only)",
        "",
        "| Model | AbsRel | delta1 | valid pixels dropped as non-positive |",
        "|---|---|---|---|",
        f"| V1-S | {sens_v1['abs_rel_mean']:.4f} | {sens_v1['delta1_mean']:.4f} | {sens_v1['nonpositive_px_fraction']:.1%} |",
        f"| V2-S | {sens_v2['abs_rel_mean']:.4f} | {sens_v2['delta1_mean']:.4f} | {sens_v2['nonpositive_px_fraction']:.1%} |",
        "",
        f"Ties on DA-2K: V1-S {da2k['v1s']['n_ties']}, V2-S {da2k['v2s']['n_ties']} (counted as incorrect).",
        "",
    ]
    if verdict:
        lines += ["## Pre-registered verdict", "", f"**{verdict['verdict']}**: {verdict['reason']}", ""]
    out.write_text("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args(argv)
    rd = Path(args.results_dir)
    da2k = json.loads((rd / "da2k" / "da2k_comparison.json").read_text())
    diode = json.loads((rd / "diode" / "diode_comparison.json").read_text())
    verdict_path = rd / "final_verdict.json"
    verdict = json.loads(verdict_path.read_text()) if verdict_path.is_file() else None

    figdir = rd / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig_headline(da2k, diode, figdir / "fig1_v2_minus_v1.png")
    fig_da2k_by_scene(da2k, figdir / "fig2_da2k_by_scene.png")
    fig_diode_by_split(diode, figdir / "fig3_diode_by_split.png")
    results_table(da2k, diode, verdict, rd / "results_table.md")
    print(f"Wrote figures to {figdir} and {rd / 'results_table.md'}")


if __name__ == "__main__":
    main()
