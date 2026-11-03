#!/usr/bin/env python3
"""
mrta_baselines.py — R1.2: non-learning game-theoretic / market MRTA baselines in the E2 environment.

Environment, rewards, start cells (R0=(0,0), R1=(4,4)) and episode length are exactly E2's
(MixedMotiveGridWorld, n_tasks=1: the task always re-spawns at (0,0)).

Planners (centralized, full information, re-plan every step):
  nearest   : single-item auction with bid = Manhattan distance to the task (Gerkey & Mataric 2004,
              greedy/SSI with one task). Equivalent to lambda = 0 below.
  balanced_l*: same auction, bid = distance + lambda * tasks already won in the episode
              (load-balancing market). lambda = 1 trades one task for one cell of travel.
The auction winner moves toward the task; the loser moves toward the free cell adjacent to the task
(it then earns the cooperation reward). Moves are one cell per step along a shortest path; the loser
yields (stays) if both planned cells coincide. A swap of two adjacent robots is legal in E2.

Execution noise: with probability eps each robot executes a uniform random action instead
(eps = 0.05 matches the frozen-policy protocol used for Nash).

Usage: python3 revision_r12/mrta_baselines.py [--episodes 100] [--seeds 0..19]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nash_depin.environment_mixed import MixedMotiveGridWorld

TASK = (0, 0)
MOVES = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3, (0, 0): 4}


def dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def step_toward(p, t):
    if p == t:
        return p
    if p[0] != t[0]:
        return (p[0] + (1 if t[0] > p[0] else -1), p[1])
    return (p[0], p[1] + (1 if t[1] > p[1] else -1))


def plan(pos, won, lam):
    bids = [dist(pos[i], TASK) + lam * won[i] for i in (0, 1)]
    w = int(np.argmin(bids))  # ties -> robot 0
    l = 1 - w
    adj = [(1, 0), (0, 1)]
    target_l = min(adj, key=lambda c: (dist(pos[l], c), c))
    nxt = [None, None]
    nxt[w] = step_toward(pos[w], TASK)
    nxt[l] = step_toward(pos[l], target_l)
    if nxt[l] == nxt[w]:
        nxt[l] = pos[l]
        if nxt[l] == nxt[w]:          # loser is standing where the winner wants to go -> swap is fine
            nxt[l] = pos[w] if dist(pos[w], pos[l]) == 1 else pos[l]
    acts = []
    for i in (0, 1):
        d = (nxt[i][0] - pos[i][0], nxt[i][1] - pos[i][1])
        acts.append(MOVES[d])
    return acts, w


def run(lam, eps, episodes, seed):
    rng = np.random.default_rng(seed)
    env = MixedMotiveGridWorld(grid_size=5, n_agents=2, n_tasks=1, max_steps=200, seed=seed)
    R = np.zeros(2); T = np.zeros(2); coll = 0; hand = 0; moves = 0; epg = []
    for _ in range(episodes):
        env.reset()
        won = [0, 0]; er = np.zeros(2); holder = None
        for _ in range(200):
            acts, _ = plan(list(env.agent_positions), won, lam)
            acts = [int(rng.integers(5)) if rng.random() < eps else a for a in acts]
            before = list(env.agent_positions)
            _, _, r, _, info = env.step(tuple(acts))
            moves += sum(before[i] != env.agent_positions[i] for i in (0, 1))
            er += r; coll += info["collisions"]
            for i in (0, 1):
                if env.agent_positions[i] == TASK:
                    won[i] += 1; T[i] += 1
                    if holder is not None and holder != i:
                        hand += 1
                    holder = i
        R += er
        epg.append(abs(er[0] - er[1]) / (2 * er.sum()) if er.sum() > 0 else np.nan)
    return {"gini": round(float(abs(R[0] - R[1]) / (2 * R.sum())), 4),
            "ep_gini_median": round(float(np.nanmedian(epg)), 4),
            "total_reward_per_ep": round(float(R.sum() / episodes), 1),
            "tasks_per_ep": (T / episodes).round(1).tolist(),
            "collisions_per_ep": round(coll / episodes, 2),
            "handovers_per_ep": round(hand / episodes, 1),
            "moves_per_ep": round(moves / episodes, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    ap.add_argument("--out", default=str(ROOT / "revision_r12/mrta_results.json"))
    a = ap.parse_args()
    out = {}
    for name, lam in [("nearest", 0.0), ("balanced_l0.01", 0.01), ("balanced_l0.1", 0.1), ("balanced_l1", 1.0)]:
        out[name] = {}
        for eps in (0.0, 0.05):
            per = [run(lam, eps, 1 if eps == 0 else a.episodes, s) for s in a.seeds]
            g = [p["gini"] for p in per]
            out[name][f"eps={eps}"] = {
                "per_seed": per, "gini_median": float(np.median(g)),
                "fair_seeds": int(sum(x < 0.15 for x in g)), "n": len(g),
                "total_reward_median": float(np.median([p["total_reward_per_ep"] for p in per])),
                "collisions_median": float(np.median([p["collisions_per_ep"] for p in per])),
                "handovers_median": float(np.median([p["handovers_per_ep"] for p in per])),
                "moves_median": float(np.median([p["moves_per_ep"] for p in per])),
                "tasks_median": np.median([p["tasks_per_ep"] for p in per], axis=0).tolist()}
            s = out[name][f"eps={eps}"]
            print(f"{name:14s} eps={eps:<4} Gini {s['gini_median']:.4f} fair {s['fair_seeds']}/{s['n']} "
                  f"reward {s['total_reward_median']:.0f} tasks {s['tasks_median']} "
                  f"coll {s['collisions_median']:.2f} hand-overs {s['handovers_median']:.1f} moves {s['moves_median']:.0f}")
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
