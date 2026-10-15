#!/usr/bin/env python3
"""
kalai_ablation.py — R1.2 / R2.3: ablation of the egalitarian (Kalai) equilibrium selection in E2.

Retrains E2 Nash Q-learning (same environment, hyperparameters, seeds and code path as
experiments/run_e2.py) changing ONLY the equilibrium-selection step, then re-evaluates the
final Q-tables as frozen policies (same protocol as reeval/reeval_frozen_e2.py, E2 start).

Variants (selection of the stage-game equilibrium used for both the Nash-Q target and the policy):
  egalitarian  : joint_nash(fast=True, equity=True)  -> 4 best-response starts, min |V0-V1|
                 (= the paper's E2; used as a regression check against qtables_all_seeds)
  utilitarian  : SAME 4 starts, max V0+V1             -> isolates the selection rule
  single       : joint_nash(fast=True, equity=False) -> 1 start (uniform) only; this is what the
                 code flag USE_EQUITY=False does (changes the candidate set AND the rule)

Usage (repository root):
  python3 revision_r12/kalai_ablation.py --variant utilitarian --seeds 0 1 2 --out revision_r12/results
  python3 revision_r12/kalai_ablation.py --variant egalitarian --seeds 0 --episodes 1000 ...
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import nash_depin.nash_agent_e2 as agent_mod
from nash_depin import nash_solver
from nash_depin.nash_solver import joint_nash, _best_response_iterate
from nash_depin.environment_mixed import MixedMotiveGridWorld
from nash_depin.fairness_metrics import compute_all_fairness
from reeval.reeval_frozen_e2 import run as frozen_run


def solver_for(variant):
    if variant == "egalitarian":
        return lambda A, B, fast=True, equity=True: joint_nash(A, B, fast=True, equity=True)
    if variant == "single":
        return lambda A, B, fast=True, equity=False: joint_nash(A, B, fast=True, equity=False)
    if variant == "utilitarian":
        def util(A, B, fast=True, equity=None):
            n = A.shape[0]
            unif = np.ones(n) / n
            e0 = np.zeros(n); e0[0] = 1.0
            e1 = np.zeros(n); e1[min(1, n - 1)] = 1.0
            cands = [_best_response_iterate(A, B, i0, i1)
                     for i0, i1 in [(unif, unif), (e0, unif), (unif, e0), (e0, e1)]]
            return max(cands, key=lambda c: c[2] + c[3])
        return util
    raise ValueError(variant)


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def train(cfg, seed, episodes, variant):
    # Patch the solver used inside NashTeamE2.learn (module-level name), nothing else changes.
    agent_mod.joint_nash = solver_for(variant)
    ec = cfg["environment"]
    env = MixedMotiveGridWorld(
        grid_size=ec["grid_size"], n_agents=ec["n_agents"], n_tasks=ec["num_tasks"],
        task_reward=ec["task_reward"], collision_penalty=ec["collision_penalty"],
        cooperation_reward=ec["cooperation_reward"], coop_distance=ec["coop_distance"],
        max_steps=ec["max_steps"], seed=seed)
    team = agent_mod.NashTeamE2(env.n_states, env.N_ACTIONS, cfg, seed=seed)
    ep_r, ep_t, ep_c = [], [], []
    for _ in range(episodes):
        state, obs = env.reset()
        cr, ct, cc, done = [0.0, 0.0], [0, 0], 0, False
        while not done:
            actions = team.act(state, obs)
            ns, nobs, rewards, done, info = env.step(actions)
            team.learn(state, actions, rewards, ns)
            for i in range(2):
                cr[i] += rewards[i]
                if info["tasks_completed"] > 0 and rewards[i] >= ec["task_reward"]:
                    ct[i] += 1
            cc += info["collisions"]
            state, obs = ns, nobs
        team.end_episode()
        ep_r.append(tuple(cr)); ep_t.append(tuple(ct)); ep_c.append(cc)
    n = cfg["experiment_control"]["eval_last_n"]
    mr = np.array(ep_r[-n:]).mean(axis=0); mt = np.array(ep_t[-n:]).mean(axis=0)
    fair = compute_all_fairness(mr.tolist(), mt.tolist())
    fair["mean_collisions"] = float(np.mean(ep_c[-n:]))
    fair["mean_tasks"] = mt.round(2).tolist()
    return team, fair


def frozen_policies(q, variant):
    sel = solver_for(variant)
    P = [np.zeros((625, 5)), np.zeros((625, 5))]
    for s in range(625):
        s0, s1, _, _ = sel(q[0][s], q[1][s])
        P[0][s], P[1][s] = s0, s1
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["egalitarian", "utilitarian", "single"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--config", default=str(ROOT / "config/e2_config.yaml"))
    ap.add_argument("--out", default=str(ROOT / "revision_r12/results"))
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    episodes = a.episodes or cfg["experiment_control"]["episodes"]
    out = Path(a.out) / a.variant; (out / "qtables").mkdir(parents=True, exist_ok=True)
    for seed in a.seeds:
        t0 = time.time()
        team, fair = train(cfg, seed, episodes, a.variant)
        q = [ag.q_table for ag in team.agents]
        files = {}
        for i in (0, 1):
            p = out / "qtables" / f"seed_{seed}_agent_{i}.npz"
            np.savez_compressed(p, q_table=q[i]); files[p.name] = sha256(p)  # same format as qtables_all_seeds
        P = frozen_policies(q, a.variant)
        ev = {"sample_eps": frozen_run(P, 0.05, "sample", "e2", seed=seed),
              "argmax_eps": frozen_run(P, 0.05, "argmax", "e2", seed=seed),
              "greedy": frozen_run(P, 0.0, "argmax", "e2", episodes=1, seed=seed)}
        rec = {"variant": a.variant, "seed": seed, "episodes": episodes, "train_last100": fair,
               "frozen_e2": ev, "qtables_sha256": files, "elapsed_s": round(time.time() - t0, 1),
               "numpy": np.__version__, "python": sys.version.split()[0]}
        if a.variant == "egalitarian":  # regression control vs the published E2 Q-tables
            ref = json.load(open(ROOT / "qtables_all_seeds/manifest.json"))["checksums"]
            rec["matches_published_qtables"] = (all(ref[k] == v for k, v in files.items())
                                                if all(k in ref for k in files) else None)  # None: seed not published
            print(f"  egalitarian control: identical to qtables_all_seeds = {rec['matches_published_qtables']}")
        (out / f"seed_{seed}.json").write_text(json.dumps(rec, indent=1))
        print(f"{a.variant} seed {seed}: train Gini {fair['gini']:.4f} tasks {fair['mean_tasks']} | "
              f"frozen sample_eps Gini {ev['sample_eps']['gini']:.4f} tasks {ev['sample_eps']['tasks_per_ep']} | "
              f"{rec['elapsed_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
