#!/usr/bin/env python3
"""compare_marl.py — Nash (egalitarian) vs QMIX / VDN / QPLEX, 20 seeds, R1.2 lab results.

Usage (repository root):  python3 revision_r12/compare_marl.py | tee revision_r12/compare_marl_lab.txt
Primary protocol: frozen final policy in E2 (100 eps, eps = 0.05). Nash samples sigma_i; MARL uses
the frozen dict that exists (the key used is printed). Train-pool (last 100 training eps) as robustness.
Tests: one-sided Mann-Whitney U (Nash Gini < baseline Gini), Holm over the 3 baselines; Cliff's delta;
Fisher exact on fair-seed counts. Jain per seed from J = 1/(1+4G^2) (exact for n = 2).
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu, fisher_exact

R = Path(__file__).resolve().parent
SRC = {
    "nash_egal": R / "results_lab_20261014_201553/egalitarian",
    "qmix": R / "results_qmix_lab_rerun/qmix",
    "vdn": R / "results_vdn_lab_rerun/vdn",
    "qplex": R / "results_qplex_lab_20261015_072008/qplex",
}
FAIR = 0.15


def frozen_block(fz):
    for k in ("sample_eps", "argmax_eps"):
        if k in fz and isinstance(fz[k], dict) and "gini" in fz[k]:
            return k, fz[k]
    for k, v in fz.items():
        if isinstance(v, dict) and "gini" in v:
            return k, v
    raise KeyError(f"no frozen dict with 'gini': {list(fz)}")


def load(d):
    recs = sorted((json.load(open(f)) for f in d.glob("seed_*.json")), key=lambda r: r["seed"])
    key, _ = frozen_block(recs[0]["frozen_e2"])
    out = {"key": key, "seeds": [r["seed"] for r in recs]}
    out["g_fz"] = np.array([frozen_block(r["frozen_e2"])[1]["gini"] for r in recs])
    out["g_tr"] = np.array([r["train_last100"]["gini"] for r in recs])
    out["rew_tr"] = np.array([r["train_last100"]["total_reward"] for r in recs])
    fzr = [frozen_block(r["frozen_e2"])[1].get("total_reward") for r in recs]
    out["rew_fz"] = np.array(fzr, float) if None not in fzr else None
    return out


def cliff(x, y):
    x, y = np.asarray(x)[:, None], np.asarray(y)[None, :]
    return float((x > y).mean() - (x < y).mean())


def iqr(a):
    return np.percentile(a, 25), np.percentile(a, 75)


def jain(g):
    return 1.0 / (1.0 + 4.0 * np.asarray(g) ** 2)


D = {k: load(v) for k, v in SRC.items()}
n = D["nash_egal"]
print(f"{'method':10s} n  key         Gini fz med [IQR]        Jain med  fair fz  Gini tr  fair tr  "
      f"sdGini  reward tr mean±sd   reward tr med  reward fz mean±sd")
for k, d in D.items():
    g, gt = d["g_fz"], d["g_tr"]
    lo, hi = iqr(g)
    rf = f"{d['rew_fz'].mean():7.1f} ± {d['rew_fz'].std(ddof=1):5.1f}" if d["rew_fz"] is not None else "   n/a"
    print(f"{k:10s} {len(g):2d} {d['key']:11s} {np.median(g):.3f} [{lo:.3f}, {hi:.3f}]  "
          f"{np.median(jain(g)):.3f}   {int((g < FAIR).sum()):2d}/{len(g)}   {np.median(gt):.3f}   "
          f"{int((gt < FAIR).sum()):2d}/{len(gt)}   {g.std(ddof=1):.4f}  "
          f"{d['rew_tr'].mean():7.1f} ± {d['rew_tr'].std(ddof=1):5.1f}   {np.median(d['rew_tr']):7.1f}   {rf}")

print("\nNash egalitarian vs baselines (frozen Gini; one-sided MWU, Holm over 3):")
raw = {}
for k in ("qmix", "vdn", "qplex"):
    raw[k] = mannwhitneyu(n["g_fz"], D[k]["g_fz"], alternative="less").pvalue
order = sorted(raw, key=raw.get)
holm, run = {}, 0.0
for i, k in enumerate(order):
    run = max(run, min(1.0, (len(order) - i) * raw[k]))
    holm[k] = run
for k in ("qmix", "vdn", "qplex"):
    b = D[k]
    fn, fb = int((n["g_fz"] < FAIR).sum()), int((b["g_fz"] < FAIR).sum())
    fp = fisher_exact([[fn, len(n["g_fz"]) - fn], [fb, len(b["g_fz"]) - fb]])[1]
    tr = mannwhitneyu(n["g_tr"], b["g_tr"], alternative="less").pvalue
    dr = (n["rew_tr"].mean() - b["rew_tr"].mean()) / b["rew_tr"].mean() * 100
    print(f"  vs {k:6s} p_raw={raw[k]:.2e}  p_holm={holm[k]:.2e}  cliff={cliff(n['g_fz'], b['g_fz']):+.2f}  "
          f"fisher(2s)={fp:.4f}  p_train(raw)={tr:.2e}  reward diff (train mean)={dr:+.2f}%")

print("\nNash monopolistic seeds (frozen Gini >= 0.15):")
for s, g, gt in zip(n["seeds"], n["g_fz"], n["g_tr"]):
    if g >= FAIR:
        print(f"  seed {s:2d}  frozen {g:.3f}  train {gt:.3f}")
print("Fair seeds:", [s for s, g in zip(n["seeds"], n["g_fz"]) if g < FAIR])
for k in ("qmix", "vdn", "qplex"):
    print(f"{k} fair seed(s):", [(s, round(float(g), 3)) for s, g in zip(D[k]["seeds"], D[k]["g_fz"]) if g < FAIR])
