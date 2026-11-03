#!/usr/bin/env python3
"""
qplex_run.py — R1.2: QPLEX in E2 (same loop, env, hyperparameters and metrics as run_e2.py for QMIX).

  python3 revision_r12/qplex_run.py --algo qplex --seeds 0 1 ... --out revision_r12/results_qplex
  python3 revision_r12/qplex_run.py --algo qmix  --seeds 0          # harness control vs e2_results/qmix
  python3 revision_r12/qplex_run.py --algo vdn   --seeds 0 ...      # VDN rerun (same call as run_e2.py)

Per seed: training metrics over the last 100 episodes (paper metric for QMIX/VDN) and a frozen
evaluation (greedy per-agent argmax, eps = 0.05, E2 start, 100 episodes; same as reeval for Nash).
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nash_depin.environment_mixed import MixedMotiveGridWorld
from nash_depin.fairness_metrics import compute_all_fairness
from baselines.qmix_baseline import QMIXTeam
from baselines.vdn_baseline import VDNTeam
from revision_r12.qplex_baseline import QPLEXTeam

torch.set_num_threads(1)


def make_env(ec, seed):
    return MixedMotiveGridWorld(grid_size=ec["grid_size"], n_agents=ec["n_agents"], n_tasks=ec["num_tasks"],
                                task_reward=ec["task_reward"], collision_penalty=ec["collision_penalty"],
                                cooperation_reward=ec["cooperation_reward"], coop_distance=ec["coop_distance"],
                                max_steps=ec["max_steps"], seed=seed)


def train(algo, cfg, seed, episodes):
    ec = cfg["environment"]; env = make_env(ec, seed)
    if algo == "vdn":
        team = VDNTeam(env.OBS_DIM, env.N_ACTIONS, cfg, seed=seed)
    else:
        Team = QPLEXTeam if algo == "qplex" else QMIXTeam
        team = Team(env.OBS_DIM, env.state_dim, env.N_ACTIONS, cfg, seed=seed)
    ep_r, ep_t, ep_c = [], [], []
    for _ in range(episodes):
        state, obs = env.reset(); sv = env.get_global_state_vec()
        cr, ct, cc, done = [0.0, 0.0], [0, 0], 0, False
        while not done:
            actions = team.act(state, obs)
            ns, nobs, rewards, done, info = env.step(actions); nsv = env.get_global_state_vec()
            if algo == "vdn":   # same call as experiments/run_e2.py
                team.learn(state, actions, rewards, ns, obs, nobs, done)
            else:
                team.learn(sv, actions, rewards, nsv, obs, nobs, done)
            for i in range(2):
                cr[i] += rewards[i]
                if info["tasks_completed"] > 0 and rewards[i] >= ec["task_reward"]:
                    ct[i] += 1
            cc += info["collisions"]; state, obs, sv = ns, nobs, nsv
        team.end_episode()
        ep_r.append(tuple(cr)); ep_t.append(tuple(ct)); ep_c.append(cc)
    n = cfg["experiment_control"]["eval_last_n"]
    mr = np.array(ep_r[-n:]).mean(axis=0); mt = np.array(ep_t[-n:]).mean(axis=0)
    fair = compute_all_fairness(mr.tolist(), mt.tolist())
    fair["mean_collisions"] = float(np.mean(ep_c[-n:])); fair["mean_tasks"] = mt.round(2).tolist()
    return team, fair


def greedy_actions(team, obs):
    with torch.no_grad():
        return tuple(int(team.q_nets[i](torch.tensor(obs[i], dtype=torch.float32)).argmax().item()) for i in (0, 1))


def frozen(team, eps, episodes, seed):
    rng = np.random.default_rng(seed)
    env = MixedMotiveGridWorld(grid_size=5, n_agents=2, n_tasks=1, max_steps=200, seed=seed)
    R = np.zeros(2); T = np.zeros(2); coll = 0; hand = 0
    for _ in range(episodes):
        _, obs = env.reset(); holder = None
        for _ in range(200):
            g = greedy_actions(team, obs)
            acts = tuple(int(rng.integers(5)) if rng.random() < eps else g[i] for i in (0, 1))
            _, obs, r, _, info = env.step(acts); R += r; coll += info["collisions"]
            for i in (0, 1):
                if env.agent_positions[i] == (0, 0):
                    T[i] += 1
                    if holder is not None and holder != i: hand += 1
                    holder = i
    return {"gini": round(float(abs(R[0] - R[1]) / (2 * R.sum())), 4), "tasks_per_ep": (T / episodes).round(1).tolist(),
            "collisions_per_ep": round(coll / episodes, 2), "handovers_per_ep": round(hand / episodes, 2),
            "total_reward_per_ep": round(float(R.sum() / episodes), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="qplex", choices=["qplex", "qmix", "vdn"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--config", default=str(ROOT / "config/e2_config.yaml"))
    ap.add_argument("--out", default=str(ROOT / "revision_r12/results_qplex"))
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); episodes = a.episodes or cfg["experiment_control"]["episodes"]
    out = Path(a.out) / a.algo; out.mkdir(parents=True, exist_ok=True)
    published = {}
    if a.algo in ("qmix", "vdn"):
        published = {r["seed"]: r["gini"] for r in json.load(open(ROOT / f"e2_results/{a.algo}/summary.json"))["per_seed"]}
    for seed in a.seeds:
        t0 = time.time()
        team, fair = train(a.algo, cfg, seed, episodes)
        rec = {"algo": a.algo, "seed": seed, "episodes": episodes, "train_last100": fair,
               "frozen_e2": {"argmax_eps": frozen(team, 0.05, 100, seed), "greedy": frozen(team, 0.0, 1, seed)},
               "elapsed_s": round(time.time() - t0, 1), "torch": torch.__version__, "numpy": np.__version__}
        if seed in published:
            rec["published_train_gini"] = published[seed]
            rec["matches_published"] = abs(published[seed] - fair["gini"]) < 1e-6
        torch.save({f"agent_{i}": team.q_nets[i].state_dict() for i in (0, 1)}, out / f"seed_{seed}_agents.pt")
        (out / f"seed_{seed}.json").write_text(json.dumps(rec, indent=1))
        fz = rec["frozen_e2"]["argmax_eps"]
        print(f"{a.algo} seed {seed}: train Gini {fair['gini']:.4f} tasks {fair['mean_tasks']} reward {fair['total_reward']:.0f} | "
              f"frozen Gini {fz['gini']:.4f} tasks {fz['tasks_per_ep']} hand-overs {fz['handovers_per_ep']} | "
              f"{rec['elapsed_s']:.0f}s" + (f" | published {published[seed]:.4f} match={rec['matches_published']}" if seed in published else ""),
              flush=True)


if __name__ == "__main__":
    main()
