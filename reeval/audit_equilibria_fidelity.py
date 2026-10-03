#!/usr/bin/env python3
"""
audit_equilibria_fidelity.py — Paper 1 correction, Sec. III-C and V-D.

(1) Exploitability of the profiles selected by the fast solver (joint_nash, fast=True,
    equity=True) at the ten final E2 Q-tables, relative to each stage game's payoff range.
(3) Task-count logging rule: share of completions omitted by the run_e2.py rule
    (completion counted only if the step reward is >= R_task), frozen Nash policies,
    E2 start, sampling sigma_i with epsilon=0.05.
(2) Probability that a policy SAMPLING from the learned mixed equilibrium reproduces the
    greedy (argmax) cell sequence of the deployed seeds 2 and 4 from the physical start,
    for 12-step (E4) and 200-step (E3) episodes; Monte Carlo.
Run from the repository root:  python3 reeval/audit_equilibria_fidelity.py
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reeval_frozen_e2 import load, policies, joint_nash, MixedMotiveGridWorld

def gap(A, B, s0, s1):
    return max(0.0, float(np.max(A @ s1) - s0 @ A @ s1), float(np.max(B.T @ s0) - s0 @ B @ s1))

def cells(P, mode, steps, rng):
    env = MixedMotiveGridWorld(grid_size=5, n_agents=2, n_tasks=1, max_steps=steps, seed=0)
    env.reset(); env.agent_positions = [(4, 0), (0, 4)]; s = env._encode_state(); out = []
    for _ in range(steps):
        a = []
        for i in (0, 1):
            if mode == "argmax": a.append(int(np.argmax(P[i][s])))
            else:
                p = np.clip(P[i][s], 0, None); a.append(int(rng.choice(5, p=p / p.sum())))
        s, *_ = env.step(tuple(a)); out.append(tuple(env.agent_positions))
    return out

if __name__ == "__main__":
    rel = []
    for seed in range(10):
        q = load(seed)
        for s in range(625):
            if np.abs(q[0][s]).sum() == 0: continue
            s0, s1, _, _ = joint_nash(q[0][s], q[1][s], fast=True, equity=True)
            rel.append(gap(q[0][s], q[1][s], s0, s1) / max(np.ptp(q[0][s]), np.ptp(q[1][s]), 1e-9))
    rel = np.array(rel)
    res = {"exploitability_relative": {"states": int(rel.size), "median": float(np.median(rel)),
           "p95": float(np.percentile(rel, 95)), "p99": float(np.percentile(rel, 99)), "max": float(rel.max())}}
    rng = np.random.default_rng(1)
    for seed in (2, 4):
        P = policies(load(seed))
        for steps, n in ((12, 4000), (200, 1000)):
            ref = cells(P, "argmax", steps, rng)
            res[f"seed{seed}_match_{steps}"] = sum(cells(P, "sample", steps, rng) == ref for _ in range(n)) / n
    for seed in range(10):
        P = policies(load(seed)); r2 = np.random.default_rng(seed)
        env = MixedMotiveGridWorld(grid_size=5, n_agents=2, n_tasks=1, max_steps=200, seed=seed)
        pos = 0; rew = 0
        for _ in range(100):
            env.reset(); s = env._encode_state()
            for _ in range(200):
                a = []
                for i in (0, 1):
                    if r2.random() < 0.05: a.append(int(r2.integers(5)))
                    else:
                        p = np.clip(P[i][s], 0, None); a.append(int(r2.choice(5, p=p / p.sum())))
                s, _, r, _, info = env.step(tuple(a))
                if info["tasks_completed"]:
                    pos += 1; rew += int(max(r) >= 10)
        res[f"seed{seed}_completions_omitted_by_logging"] = round(1 - rew / pos, 4)
    (Path(__file__).parent / "results_audit.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
