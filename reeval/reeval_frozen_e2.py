#!/usr/bin/env python3
"""
reeval_frozen_e2.py — Paper 1 (IEEE Access) correction, Sec. IV-B "Mechanism check".

Re-evaluates the ten final E2 Nash Q-tables as FROZEN policies (no updates) in the
E2 environment (MixedMotiveGridWorld, n_tasks=1, task at (0,0)), 100 episodes/seed.

Behaviour policies:
  greedy      : argmax sigma_i, epsilon = 0 (deployment policy)
  sample_eps  : sample sigma_i w.p. 1-eps, uniform w.p. eps (training behaviour), eps = 0.05
  argmax_eps  : argmax sigma_i w.p. 1-eps, uniform w.p. eps, eps = 0.05
sigma_i comes from joint_nash(fast=True, equity=True), as in training.

Conditions: E2 start (R0=(0,0), R1=(4,4)), physical start (R0=(4,0), R1=(0,4)),
and Q-tables mirrored about the grid diagonal (transpose test).

Run from the repository root:  python3 reeval/reeval_frozen_e2.py
Outputs: reeval/results_reeval.json and a summary on stdout.
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nash_depin.environment_mixed import MixedMotiveGridWorld
from nash_depin.nash_solver import joint_nash

QDIR = ROOT / "qtables_all_seeds"
T = [2, 3, 0, 1, 4]                      # transpose: up<->left, down<->right, stay
STARTS = {"e2": [(0, 0), (4, 4)], "physical": [(4, 0), (0, 4)]}

def dec(s): a, b = divmod(s, 25); return a // 5, a % 5, b // 5, b % 5
def enc(r0, c0, r1, c1): return (r0 * 5 + c0) * 25 + (r1 * 5 + c1)

def load(seed, transpose=False):
    q = [np.load(QDIR / f"seed_{seed}_agent_{i}.npz")["q_table"] for i in (0, 1)]
    if not transpose: return q
    qt = [np.zeros_like(x) for x in q]
    for s in range(625):
        r0, c0, r1, c1 = dec(s); st = enc(c0, r0, c1, r1)
        for a0 in range(5):
            for a1 in range(5):
                for i in (0, 1): qt[i][st, T[a0], T[a1]] = q[i][s, a0, a1]
    return qt

def policies(q):
    P = [np.zeros((625, 5)), np.zeros((625, 5))]
    for s in range(625):
        s0, s1, _, _ = joint_nash(q[0][s], q[1][s], fast=True, equity=True)
        P[0][s], P[1][s] = s0, s1
    return P

def run(P, eps, mode, start, episodes=100, seed=0, steps=200):
    rng = np.random.default_rng(seed)
    env = MixedMotiveGridWorld(grid_size=5, n_agents=2, n_tasks=1, max_steps=steps, seed=seed)
    R = np.zeros(2); tasks = np.zeros(2); coll = 0; epg = []
    for _ in range(episodes):
        env.reset(); env.agent_positions = list(STARTS[start]); s = env._encode_state()
        er = np.zeros(2)
        for _ in range(steps):
            acts = []
            for i in (0, 1):
                if rng.random() < eps: acts.append(int(rng.integers(5)))
                elif mode == "argmax": acts.append(int(np.argmax(P[i][s])))
                else:
                    p = np.clip(P[i][s], 0, None); acts.append(int(rng.choice(5, p=p / p.sum())))
            s, _, r, _, info = env.step(tuple(acts))
            er += r; coll += info["collisions"]
            tasks += np.array([pos == (0, 0) for pos in env.agent_positions])  # task fixed at (0,0)
        R += er; epg.append(abs(er[0] - er[1]) / (2 * er.sum()) if er.sum() > 0 else np.nan)
    return {"gini": round(float(abs(R[0] - R[1]) / (2 * R.sum())), 4),
            "ep_gini_median": round(float(np.nanmedian(epg)), 3),
            "tasks_per_ep": (tasks / episodes).round(1).tolist(),
            "collisions_per_ep": round(coll / episodes, 2),
            "final_cells": [list(p) for p in env.agent_positions]}

if __name__ == "__main__":
    out = {}
    for cond, start, tr in [("e2", "e2", False), ("e2_transposed", "e2", True), ("physical", "physical", False)]:
        out[cond] = {}
        for seed in range(10):
            P = policies(load(seed, tr))
            out[cond][seed] = {"greedy": run(P, 0.0, "argmax", start, episodes=1, seed=seed),
                               "sample_eps": run(P, 0.05, "sample", start, seed=seed),
                               "argmax_eps": run(P, 0.05, "argmax", start, seed=seed)}
            print(cond, seed, json.dumps(out[cond][seed]), flush=True)
    (Path(__file__).parent / "results_reeval.json").write_text(json.dumps(out, indent=1))
    print("\nSUMMARY (fair = pooled Gini < 0.15)")
    for cond in out:
        for m in ("sample_eps", "argmax_eps"):
            g = [out[cond][s][m]["gini"] for s in range(10)]
            print(f"{cond:14s} {m:10s} median={np.median(g):.4f} fair={[s for s in range(10) if g[s] < 0.15]}")
