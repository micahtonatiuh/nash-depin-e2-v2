"""
analyze_e2.py — Figures and descriptive comparison for E2.
Nash-DePIN.

Reads the summary.json of the three algorithms (nash, qmix, vdn) and produces
the Gini boxplot, the Gini/Jain bar chart, and a comparison table.

IMPORTANT — statistical scope:
The parametric statistics computed here (two-sample t-test, Cohen's d, mean +/-
std) are DESCRIPTIVE ONLY and are NOT the statistics reported in the paper.
The Nash Gini distribution is bimodal (a subset of seeds converges to stable
monopoly equilibria), which violates the normality assumption of the t-test and
makes the mean an unstable summary. The primary analysis is
experiments/analyze_e2_stats_v2.py, which uses the MEDIAN, Mann-Whitney U with
Holm correction, and Cliff's delta. Where the two disagree, the v2 analysis
governs.

Usage:
    python3 scripts/analyze_e2.py --results e2_results/ --output e2_results/plots/
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def cohens_d(a, b):
    """Cohen's d effect size between two samples."""
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def load_summary(results_dir, algo):
    path = Path(results_dir) / algo / "summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    algos = ["nash", "qmix", "vdn"]
    labels = {"nash": "Nash Q-Learning", "qmix": "QMIX", "vdn": "VDN"}
    colors = {"nash": "#1D9E75", "qmix": "#D85A30", "vdn": "#BA7517"}

    data = {a: load_summary(args.results, a) for a in algos}
    missing = [a for a in algos if data[a] is None]
    if missing:
        print(f"Missing results for: {missing}")
        print("Run run_e2.py for each missing algorithm first.")
        sys.exit(1)

    # Per-seed Gini and Jain
    ginis = {a: [s["gini"] for s in data[a]["per_seed"]] for a in algos}
    jains = {a: [s["jain"] for s in data[a]["per_seed"]] for a in algos}

    # ----------------------------------------------------------------
    # Descriptive statistics (see the module docstring: NOT the primary
    # analysis reported in the paper)
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("E2 DESCRIPTIVE SUMMARY — Fairness (Gini)")
    print("  Primary analysis: experiments/analyze_e2_stats_v2.py (median,")
    print("  Mann-Whitney U + Holm, Cliff's delta). The figures below are")
    print("  parametric and descriptive only.")
    print("=" * 60)
    for a in algos:
        print(f"  {labels[a]:18s}: Gini {np.mean(ginis[a]):.4f} +/- {np.std(ginis[a]):.4f} "
              f"| median {np.median(ginis[a]):.4f} | Jain {np.mean(jains[a]):.4f}")

    print("\n  Comparisons (two-sample t-test, descriptive only):")
    stats_rows = []
    for baseline in ["qmix", "vdn"]:
        t_stat, p_val = stats.ttest_ind(ginis["nash"], ginis[baseline])
        d = cohens_d(ginis[baseline], ginis["nash"])  # positive if baseline > nash
        sig = "significant" if p_val < 0.05 else "not significant"
        print(f"    Nash vs {labels[baseline]:6s}: p={p_val:.5f} | Cohen's d={d:.3f} | {sig}")
        stats_rows.append({
            "comparison": f"Nash vs {labels[baseline]}",
            "p_value": round(float(p_val), 6),
            "cohens_d": round(float(d), 4),
            "significant": bool(p_val < 0.05),
        })

    # ----------------------------------------------------------------
    # Preregistered hypothesis check (mean-based; reported in the paper via
    # the median in analyze_e2_stats_v2.py)
    # ----------------------------------------------------------------
    nash_ok = np.mean(ginis["nash"]) < 0.15
    base_ok = all(np.mean(ginis[b]) > 0.30 for b in ["qmix", "vdn"])
    sig_ok = all(r["significant"] for r in stats_rows)
    confirmed = nash_ok and base_ok and sig_ok

    print("\n  HYPOTHESIS CHECK (mean-based):")
    print(f"    Nash mean Gini < 0.15:   {'yes' if nash_ok else 'no'} ({np.mean(ginis['nash']):.4f})")
    print(f"    Baselines Gini > 0.30:   {'yes' if base_ok else 'no'}")
    print(f"    p < 0.05 (both):         {'yes' if sig_ok else 'no'}")
    print(f"    Mean-based verdict: {'supported' if confirmed else 'inconclusive'}")
    print("    Note: the Nash mean sits close to the 0.15 threshold and is")
    print("    sensitive to single seeds; see the leave-one-out check in")
    print("    e2_stats_report.json.")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Figure 1: Gini boxplot
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    box_data = [ginis[a] for a in algos]
    bp = ax.boxplot(box_data, labels=[labels[a] for a in algos],
                    patch_artist=True, widths=0.5, showmeans=True)
    for patch, a in zip(bp["boxes"], algos):
        patch.set_facecolor(colors[a]); patch.set_alpha(0.55)
    for median in bp["medians"]:
        median.set_color("black"); median.set_linewidth(1.2)

    ax.axhline(0.15, color="#1D9E75", linestyle="--", linewidth=1, label="Nash target (0.15)")
    ax.axhline(0.30, color="#D85A30", linestyle=":", linewidth=1, label="Baseline threshold (0.30)")
    ax.set_ylabel("Gini coefficient (lower = fairer)", fontsize=11)
    ax.set_title("E2 — Fairness: Nash Q-Learning vs QMIX/VDN", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(0.5, max(max(g) for g in box_data) * 1.1))

    plt.tight_layout()
    out1 = Path(args.output) / "e2_gini_boxplot.pdf"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.savefig(str(out1).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out1}")

    # ----------------------------------------------------------------
    # Figure 2: mean Gini +/- std and Jain index
    # ----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(algos))

    means_g = [np.mean(ginis[a]) for a in algos]
    stds_g = [np.std(ginis[a]) for a in algos]
    ax1.bar(x, means_g, yerr=stds_g, capsize=5,
            color=[colors[a] for a in algos], alpha=0.7)
    ax1.axhline(0.15, color="#1D9E75", linestyle="--", linewidth=1)
    ax1.axhline(0.30, color="#D85A30", linestyle=":", linewidth=1)
    ax1.set_xticks(x); ax1.set_xticklabels([labels[a] for a in algos], fontsize=9)
    ax1.set_ylabel("Gini"); ax1.set_title("Gini coefficient", fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")

    means_j = [np.mean(jains[a]) for a in algos]
    stds_j = [np.std(jains[a]) for a in algos]
    ax2.bar(x, means_j, yerr=stds_j, capsize=5,
            color=[colors[a] for a in algos], alpha=0.7)
    ax2.axhline(0.90, color="#1D9E75", linestyle="--", linewidth=1, label="Jain target (0.90)")
    ax2.set_xticks(x); ax2.set_xticklabels([labels[a] for a in algos], fontsize=9)
    ax2.set_ylabel("Jain's Index"); ax2.set_title("Jain's Fairness Index", fontweight="bold")
    ax2.set_ylim(0.5, 1.02); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out2 = Path(args.output) / "e2_fairness_bars.pdf"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.savefig(str(out2).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}")

    # ----------------------------------------------------------------
    # Comparison table (CSV)
    # ----------------------------------------------------------------
    csv_path = Path(args.output) / "e2_comparison_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Algorithm", "Gini (mean)", "Gini (std)",
                    "Jain (mean)", "Total reward", "Task balance"])
        for a in algos:
            d = data[a]
            w.writerow([labels[a], round(d["gini_mean"], 4), round(d["gini_std"], 4),
                        round(d["jain_mean"], 4), round(d["total_reward_mean"], 1),
                        round(d["task_balance_mean"], 3)])
    print(f"Saved: {csv_path}")

    # Save the descriptive statistics
    stats_path = Path(args.output) / "e2_statistics.json"
    with open(stats_path, "w") as f:
        json.dump({
            "gini_means": {a: float(np.mean(ginis[a])) for a in algos},
            "gini_stds": {a: float(np.std(ginis[a])) for a in algos},
            "comparisons": stats_rows,
            "hypothesis_confirmed": bool(confirmed),
        }, f, indent=2)
    print(f"Saved: {stats_path}")


if __name__ == "__main__":
    main()
