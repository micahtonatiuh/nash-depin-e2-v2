#!/usr/bin/env python3
"""summarize.py — aggregates revision_r12/results/<variant>/seed_*.json into a table + JSON.

Usage (repository root):  python3 revision_r12/summarize.py [--results revision_r12/results]
Fair = Gini < 0.15 (same criterion as the paper). Tests: two-sided Mann-Whitney U vs egalitarian.
"""
import argparse, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def mwu(x, y):
    try:
        from scipy.stats import mannwhitneyu
        return float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "revision_r12/results"))
    a = ap.parse_args()
    R = Path(a.results); table = {}
    for vdir in sorted(p for p in R.iterdir() if p.is_dir()):
        recs = [json.load(open(f)) for f in sorted(vdir.glob("seed_*.json"))]
        if not recs: continue
        recs.sort(key=lambda r: r["seed"])
        tg = [r["train_last100"]["gini"] for r in recs]
        fg = [r["frozen_e2"]["sample_eps"]["gini"] for r in recs]
        fa = [r["frozen_e2"]["argmax_eps"]["gini"] for r in recs]
        tot = [r["train_last100"]["total_reward"] for r in recs]
        col = [r["train_last100"]["mean_collisions"] for r in recs]
        tasks = [sum(r["train_last100"]["mean_tasks"]) for r in recs]
        table[vdir.name] = {
            "seeds": [r["seed"] for r in recs],
            "train_gini": tg, "frozen_sample_gini": fg, "frozen_argmax_gini": fa,
            "train_gini_median": float(np.median(tg)), "frozen_sample_gini_median": float(np.median(fg)),
            "fair_train": [r["seed"] for r, g in zip(recs, tg) if g < 0.15],
            "fair_frozen_sample": [r["seed"] for r, g in zip(recs, fg) if g < 0.15],
            "total_reward_median": float(np.median(tot)), "tasks_per_ep_median": float(np.median(tasks)),
            "collisions_median": float(np.median(col)),
            "greedy_tasks": [r["frozen_e2"]["greedy"]["tasks_per_ep"] for r in recs],
            "control_matches_published": [r.get("matches_published_qtables") for r in recs]
                                          if vdir.name == "egalitarian" else None,
        }
    base = table.get("egalitarian")
    for v, t in table.items():
        if base and v != "egalitarian":
            t["p_train_gini_vs_egalitarian"] = mwu(t["train_gini"], base["train_gini"])
            t["p_frozen_gini_vs_egalitarian"] = mwu(t["frozen_sample_gini"], base["frozen_sample_gini"])
    (R / "summary_r12.json").write_text(json.dumps(table, indent=1))
    print(f"{'variant':12s} {'n':>2s} {'Gini train':>10s} {'Gini frozen':>11s} {'fair tr':>7s} {'fair fz':>7s} "
          f"{'reward':>7s} {'tasks':>6s} {'coll':>5s}  p(train) p(frozen)")
    for v, t in table.items():
        print(f"{v:12s} {len(t['seeds']):2d} {t['train_gini_median']:10.4f} {t['frozen_sample_gini_median']:11.4f} "
              f"{len(t['fair_train']):7d} {len(t['fair_frozen_sample']):7d} {t['total_reward_median']:7.0f} "
              f"{t['tasks_per_ep_median']:6.1f} {t['collisions_median']:5.2f}  "
              f"{t.get('p_train_gini_vs_egalitarian')}  {t.get('p_frozen_gini_vs_egalitarian')}")
    if base and base["control_matches_published"]:
        print("egalitarian control identical to qtables_all_seeds:", base["control_matches_published"])


if __name__ == "__main__":
    main()
