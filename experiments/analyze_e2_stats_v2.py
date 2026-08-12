#!/usr/bin/env python3
"""
analyze_e2_stats_v2.py — Primary statistical analysis for E2.

This is the analysis reported in the paper. It reads the official summary.json
files (structure: {"per_seed": [{"gini","jain","seed"}, ...]}) and does not
modify them.

Methods: Mann-Whitney U (one-sided, Nash < baseline) + Holm correction +
Cliff's delta. Primary statistic: MEDIAN, not the mean, because the Nash Gini
distribution is bimodal (a subset of seeds converges to stable monopoly
equilibria; see e2_convergence_validation/).

Usage (from the repository root):
    python3 experiments/analyze_e2_stats_v2.py --results_dir e2_results/
Output: e2_stats_report.json + a console table.
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy import stats

FAIR_THRESHOLD = 0.15  # preregistered


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum(x > y for x in a for y in b)
    lt = sum(x < y for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def cliffs_magnitude(d):
    ad = abs(d)
    if ad < 0.147: return "negligible"
    if ad < 0.33:  return "small"
    if ad < 0.474: return "medium"
    return "large"


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj, running = np.empty(m), 0.0
    for rank, idx in enumerate(order):
        p = min((m - rank) * pvals[idx], 1.0)
        running = max(running, p)
        adj[idx] = running
    return adj


def load(results_dir, algo):
    d = json.load(open(Path(results_dir) / algo / "summary.json"))
    rows = sorted(d["per_seed"], key=lambda r: r["seed"])
    return {
        "gini": [r["gini"] for r in rows],
        "jain": [r["jain"] for r in rows],
        "seeds": [r["seed"] for r in rows],
    }


def describe(v):
    v = np.asarray(v)
    return {
        "n": len(v),
        "median": round(float(np.median(v)), 4),       # PRIMARY STATISTIC
        "iqr": [round(float(np.percentile(v, 25)), 4),
                round(float(np.percentile(v, 75)), 4)],
        "mean": round(float(v.mean()), 4),
        "std": round(float(v.std(ddof=1)), 4),
        "min": round(float(v.min()), 4),
        "max": round(float(v.max()), 4),
        "n_fair": int((v < FAIR_THRESHOLD).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="e2_results/")
    args = ap.parse_args()

    data = {a: load(args.results_dir, a) for a in ("nash", "qmix", "vdn")}

    report = {"fair_threshold": FAIR_THRESHOLD,
              "gini_descriptive": {a: describe(data[a]["gini"]) for a in data},
              "jain_descriptive": {a: describe(data[a]["jain"]) for a in data},
              "comparisons": {}}

    pairs = [("nash", "qmix"), ("nash", "vdn")]
    raw_p = []
    for a, b in pairs:
        u, p = stats.mannwhitneyu(data[a]["gini"], data[b]["gini"], alternative="less")
        d = cliffs_delta(data[a]["gini"], data[b]["gini"])
        raw_p.append(p)
        report["comparisons"][f"{a}_vs_{b}_gini"] = {
            "test": "Mann-Whitney U (one-sided: nash < baseline)",
            "U": float(u), "p_raw": float(p),
            "cliffs_delta": round(d, 3), "magnitude": cliffs_magnitude(d),
        }
    for (a, b), p_adj in zip(pairs, holm(raw_p)):
        c = report["comparisons"][f"{a}_vs_{b}_gini"]
        c["p_holm"] = float(p_adj)
        c["significant_0.05"] = bool(p_adj < 0.05)

    # Fragility of the Nash mean against the threshold (leave-one-out)
    nash = np.asarray(data["nash"]["gini"])
    loo = [round(float(np.delete(nash, i).mean()), 4) for i in range(len(nash))]
    report["nash_mean_leave_one_out"] = {
        "values": loo,
        "all_below_threshold": bool(all(m < FAIR_THRESHOLD for m in loo)),
        "note": "Mean is fragile w.r.t. the threshold; the MEDIAN is reported as primary.",
    }

    Path("e2_stats_report.json").write_text(json.dumps(report, indent=2))

    print("\n=== GINI (lower = fairer) ===")
    print(f"{'algo':6} {'n':>3} {'median':>8} {'IQR':>18} {'mean+/-std':>16} {'fair':>6}")
    for a in ("nash", "qmix", "vdn"):
        d = report["gini_descriptive"][a]
        print(f"{a:6} {d['n']:>3} {d['median']:>8} {str(d['iqr']):>18} "
              f"{d['mean']:>7}+/-{d['std']:<7} {d['n_fair']:>3}/{d['n']}")

    print("\n=== JAIN (higher = fairer) ===")
    for a in ("nash", "qmix", "vdn"):
        d = report["jain_descriptive"][a]
        print(f"{a:6} median={d['median']:.4f}  mean={d['mean']:.4f}+/-{d['std']:.4f}")

    print("\n=== COMPARISONS (Mann-Whitney U, Holm-corrected) ===")
    for k, c in report["comparisons"].items():
        print(f"{k}: U={c['U']:.1f}  p_raw={c['p_raw']:.2e}  p_holm={c['p_holm']:.2e}  "
              f"delta={c['cliffs_delta']} ({c['magnitude']})  sig={c['significant_0.05']}")

    print(f"\nLeave-one-out Nash mean < {FAIR_THRESHOLD}: "
          f"{report['nash_mean_leave_one_out']['all_below_threshold']}  "
          f"(hence the median is the primary statistic)")
    print("\nFull report: e2_stats_report.json")


if __name__ == "__main__":
    main()
