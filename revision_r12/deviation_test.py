#!/usr/bin/env python3
"""
deviation_test.py — incentive to deviate unilaterally ("refuse to yield": a robot standing on the task
stays there; otherwise it follows its prescribed policy). The other robot keeps its prescribed policy.
Gain = deviator's mean episode reward (deviating) - its reward when complying, same RNG stream.
Policies: balanced auction (lambda=0.01) and frozen egalitarian Nash (published Q-tables, sample_eps).
E2 environment and start, eps = 0.05, 100 episodes per seed.
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from nash_depin.environment_mixed import MixedMotiveGridWorld
from reeval.reeval_frozen_e2 import load, policies
from revision_r12.mrta_baselines import plan, TASK

def episode_rewards(kind, P, dev, seed, eps=0.05, episodes=100, lam=0.01):
    rng = np.random.default_rng(seed)
    env = MixedMotiveGridWorld(grid_size=5, n_agents=2, n_tasks=1, max_steps=200, seed=seed)
    R = np.zeros(2); T = np.zeros(2)
    for _ in range(episodes):
        env.reset(); s = env._encode_state(); won = [0, 0]
        for _ in range(200):
            if kind == "auction":
                acts, _ = plan(list(env.agent_positions), won, lam)
            else:
                acts = []
                for i in (0, 1):
                    p = np.clip(P[i][s], 0, None); acts.append(int(rng.choice(5, p=p / p.sum())))
            noise = [rng.random() < eps for _ in (0, 1)]; rnd = [int(rng.integers(5)) for _ in (0, 1)]
            acts = [rnd[i] if noise[i] else acts[i] for i in (0, 1)]
            if dev is not None and env.agent_positions[dev] == TASK:
                acts[dev] = 4                      # refuse to yield
            s, _, r, _, _ = env.step(tuple(acts)); R += r
            for i in (0, 1):
                if env.agent_positions[i] == TASK: won[i] += 1; T[i] += 1
    return R / episodes, T / episodes

out = {"auction_l0.01": [], "nash_egalitarian": []}
for seed in range(10):
    for kind, key in (("auction", "auction_l0.01"), ("nash", "nash_egalitarian")):
        P = policies(load(seed)) if kind == "nash" else None
        base, tb = episode_rewards(kind, P, None, seed)
        row = {"seed": seed, "comply_reward": base.round(1).tolist(), "comply_tasks": tb.round(1).tolist()}
        for d in (0, 1):
            rd, td = episode_rewards(kind, P, d, seed)
            row[f"gain_r{d}"] = round(float(rd[d] - base[d]), 1)
            row[f"gain_r{d}_pct"] = round(float(100 * (rd[d] - base[d]) / base[d]), 1)
            row[f"tasks_dev{d}"] = td.round(1).tolist()
        out[key].append(row); print(key, row, flush=True)
(ROOT / "revision_r12/deviation_results.json").write_text(json.dumps(out, indent=1))
for k, rows in out.items():
    g = [max(r["gain_r0_pct"], r["gain_r1_pct"]) for r in rows]
    print(f"{k}: best unilateral gain per seed (%) median {np.median(g):.1f}, range {min(g):.1f}..{max(g):.1f}")
