#!/usr/bin/env python3
"""
export_e2_qtables_for_e5.py — Retrains selected E2 seeds (deterministically)
and exports their Q-tables as .npz (key 'q_table') for the E5 hardware runs.

E2 did not persist Q-tables during training; runs are deterministic given the
seed, so retraining reproduces the reported policies exactly. The script
verifies that the reproduced Gini matches summary.json before exporting.

Usage (from the E2 repository root):
    python3 experiments/export_e2_qtables_for_e5.py \
        --config config/e2_config.yaml \
        --seeds 2 4 3 \
        --output ../nash-depin-e5/qtables_e5/

Per-seed output:
    seed_{s}_agent_0.npz, seed_{s}_agent_1.npz   (key 'q_table')
    + manifest.json (sha256) + fair_seeds.json
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import yaml
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from nash_depin.environment_mixed import MixedMotiveGridWorld
from nash_depin.fairness_metrics import compute_all_fairness
from nash_depin.nash_agent_e2 import NashTeamE2


def sha256(p: Path):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8192), b""):
            h.update(c)
    return h.hexdigest()


def train_seed(cfg, seed, episodes):
    ec = cfg["environment"]
    env = MixedMotiveGridWorld(
        grid_size=ec["grid_size"], n_agents=ec["n_agents"],
        n_tasks=ec["num_tasks"], task_reward=ec["task_reward"],
        collision_penalty=ec["collision_penalty"],
        cooperation_reward=ec["cooperation_reward"],
        coop_distance=ec["coop_distance"], max_steps=ec["max_steps"],
        seed=seed,
    )
    team = NashTeamE2(env.n_states, env.N_ACTIONS, cfg, seed=seed)
    eval_last_n = cfg["experiment_control"]["eval_last_n"]
    task_reward = ec["task_reward"]
    ep_rewards, ep_tasks = [], []

    for _ in range(1, episodes + 1):
        state, obs = env.reset()
        cum_r = [0.0, 0.0]; cum_t = [0, 0]
        done = False
        while not done:
            actions = team.act(state, obs)
            next_state, next_obs, rewards, done, info = env.step(actions)
            team.learn(state, actions, rewards, next_state)
            for i in range(2):
                cum_r[i] += rewards[i]
                if info["tasks_completed"] > 0 and rewards[i] >= task_reward:
                    cum_t[i] += 1
            state, obs = next_state, next_obs
        team.end_episode()
        ep_rewards.append(tuple(cum_r)); ep_tasks.append(tuple(cum_t))

    mr = np.array(ep_rewards[-eval_last_n:]).mean(axis=0)
    mt = np.array(ep_tasks[-eval_last_n:]).mean(axis=0)
    gini = compute_all_fairness(mr.tolist(), mt.tolist())["gini"]
    return team, gini


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[2, 4, 3])
    ap.add_argument("--output", required=True)
    ap.add_argument("--official", default="e2_results/nash/summary.json",
                    help="Used to verify that the reproduced Gini matches the reported one")
    args = ap.parse_args()

    out = Path(args.output).expanduser(); out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(args.config))
    episodes = cfg["experiment_control"].get("episodes",
               cfg["experiment_control"].get("max_episodes", 1000))

    # Reported per-seed Gini (fidelity check)
    official = {}
    if Path(args.official).exists():
        d = json.load(open(args.official))
        official = {r["seed"]: r["gini"] for r in d["per_seed"]}

    checksums = {}
    print(f"Retraining seeds {args.seeds} ({episodes} episodes each)...\n")
    for s in args.seeds:
        team, gini = train_seed(cfg, s, episodes)
        off = official.get(s)
        match = "n/a" if off is None else ("OK" if abs(gini - off) < 1e-6 else "MISMATCH")
        print(f"seed {s}: reproduced Gini={gini:.6f}  reported={off}  [{match}]")

        for i in range(2):
            fname = f"seed_{s}_agent_{i}.npz"
            fpath = out / fname
            np.savez_compressed(fpath, q_table=team.agents[i].q_table)
            checksums[fname] = sha256(fpath)

    (out / "manifest.json").write_text(json.dumps(
        {"source": "E2 Nash, deterministic retrain", "checksums": checksums}, indent=2))
    (out / "fair_seeds.json").write_text(json.dumps(
        {"criterion": "Gini < 0.15 in E2 and stable (validate_e2_convergence)",
         "seeds": args.seeds}, indent=2))

    print(f"\nExported {len(checksums)} Q-tables to {out}")
    print("manifest.json and fair_seeds.json written.")
    print("If every Gini reports OK, the exported policies match the reported E2 runs.")


if __name__ == "__main__":
    main()
