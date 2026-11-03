"""
team_q_no_igm.py — R1.1 empirical check of Proposition 1 (Paper 1 revision).

Centralized tabular team learner WITHOUT any factorization or monotonicity:
a single Q(s, a0, a1) over the full joint action, trained on the team reward
r_tot = r0 + r1 (Friend-Q / joint-action learner on the sum). If monopoly
were the strict team optimum, this learner should monopolize in every seed;
Proposition 1 predicts instead that the allocation is a tie-break.

Protocol identical to E2 (run_e2.py): same env and config, alpha=0.1,
gamma=0.95, per-agent eps-greedy exploration 1.0 -> 0.05 (x0.995/episode),
1000 episodes x 200 steps, seeds 0-9 (np.random.seed(seed), env seed=seed),
fairness via the repo's compute_all_fairness on per-agent rewards averaged
over the last 100 episodes. Greedy ties broken uniformly at random.
Also reports a frozen greedy rollout (eps=0) and a frozen eps=0.05 re-eval.

Original code is not modified; the repo is only imported.
Usage: python team_q_no_igm.py --repo /path/to/nash-depin-e2-v2 [--seeds 0-9]
"""
import argparse, json, sys
import numpy as np
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--seeds", default="0-9")
ap.add_argument("--config", default=None, help="path to e2_config.yaml (default: <repo>/config/e2_config.yaml)")
ap.add_argument("--out", default="team_q_no_igm_results.json")
ap.add_argument("--episodes", type=int, default=None, help="override (sensitivity); default = E2 config")
args = ap.parse_args()
sys.path.insert(0, args.repo)
from nash_depin.environment_mixed import MixedMotiveGridWorld  # noqa: E402
from nash_depin.fairness_metrics import compute_all_fairness  # noqa: E402

cfg = yaml.safe_load(open(args.config or f"{args.repo}/config/e2_config.yaml"))
E, L, C = cfg["environment"], cfg["learning"], cfg["experiment_control"]
lo, hi = map(int, args.seeds.split("-"))
N_EP = args.episodes or C["episodes"]
NA = 5


def make_env(seed):
    return MixedMotiveGridWorld(
        grid_size=E["grid_size"], n_agents=2, n_tasks=E["num_tasks"],
        task_reward=E["task_reward"], collision_penalty=E["collision_penalty"],
        cooperation_reward=E["cooperation_reward"], coop_distance=E["coop_distance"],
        max_steps=E["max_steps"], seed=seed)


def greedy(Q, s):
    q = Q[s]; m = q.max()
    idx = np.flatnonzero(q >= m - 1e-9)
    j = idx[np.random.randint(len(idx))]
    return divmod(int(j), NA)


def act(Q, s, eps):
    a0, a1 = greedy(Q, s)
    if np.random.random() < eps: a0 = np.random.randint(NA)
    if np.random.random() < eps: a1 = np.random.randint(NA)
    return a0, a1


def episode(env, Q, eps, learn):
    s, _ = env.reset(); R = [0.0, 0.0]; T = [0, 0]; done = False
    while not done:
        a = act(Q, s, eps)
        s2, _, r, done, info = env.step(a)
        if learn:
            target = sum(r) + L["gamma"] * Q[s2].max()
            j = a[0] * NA + a[1]
            Q[s, j] += L["alpha"] * (target - Q[s, j])
        for i in range(2):
            R[i] += r[i]
            if info["tasks_completed"] and r[i] >= E["task_reward"]: T[i] += 1
        s = s2
    return R, T


results = []
for seed in range(lo, hi + 1):
    np.random.seed(seed); env = make_env(seed)
    Q = np.zeros((env.n_states, NA * NA)); eps = L["epsilon_start"]
    hist_R, hist_T = [], []
    for _ in range(N_EP):
        R, T = episode(env, Q, eps, True)
        hist_R.append(R); hist_T.append(T)
        eps = max(L["epsilon_min"], eps * L["epsilon_decay"])
    mR = np.mean(hist_R[-C["eval_last_n"]:], 0); mT = np.mean(hist_T[-C["eval_last_n"]:], 0)
    fair = compute_all_fairness(mR.tolist(), mT.tolist())
    # frozen re-evaluation (no learning)
    gR, gT = episode(make_env(seed), Q, 0.0, False)
    fz = [episode(env, Q, L["epsilon_min"], False) for _ in range(100)]
    fR = np.mean([x[0] for x in fz], 0); fT = np.mean([x[1] for x in fz], 0)
    row = dict(seed=seed, gini=fair["gini"], jain=fair["jain"], mean_R=mR.tolist(),
               mean_tasks=mT.tolist(), total_reward=float(mR.sum()),
               top_task_share=float(mT.max() / max(mT.sum(), 1e-9)),
               greedy_R=gR, greedy_tasks=gT,
               frozen_eps_gini=compute_all_fairness(fR.tolist(), fT.tolist())["gini"])
    results.append(row)
    print(f"seed {seed}: Gini {row['gini']:.3f}  Jain {row['jain']:.3f}  "
          f"tasks {np.round(mT,1)}  top share {row['top_task_share']:.2f}  "
          f"total R {row['total_reward']:.1f}  | greedy tasks {gT}  "
          f"| frozen eps Gini {row['frozen_eps_gini']:.3f}", flush=True)

g = np.array([r["gini"] for r in results])
summ = dict(episodes=N_EP, n=len(g), gini_median=float(np.median(g)),
            gini_iqr=[float(np.percentile(g, 25)), float(np.percentile(g, 75))],
            fair_seeds=[r["seed"] for r in results if r["gini"] < 0.15],
            greedy_monopoly_seeds=[r["seed"] for r in results if min(r["greedy_tasks"]) == 0],
            total_reward_mean=float(np.mean([r["total_reward"] for r in results])),
            total_reward_std=float(np.std([r["total_reward"] for r in results])))
print("\nSUMMARY", json.dumps(summ, indent=1))
json.dump(dict(summary=summ, per_seed=results), open(args.out, "w"), indent=2)
