#!/usr/bin/env python3
"""r16_s3_deviation.py -- R1.6 diagnostic: is the S3 Nash outcome an equilibrium of the REAL game?

For each S3 Nash seed, robot j keeps its learned policy (sigma_j sampled, eps = EPS_EVAL, exactly as
in r16_scalability.evaluate) and robot i learns a single-agent best response by tabular Q-learning
in the true S3 environment. Two initializations of the best responder:
  warm: Q_i(s,a) = sum_b sigma_j(b|s) Q_i^Nash(s,a,b)   (its own learned beliefs)
  cold: ~0 (tiny noise for tie-breaking)
Then both profiles are evaluated over EVAL_EPISODES with the same env seed and eps as the paper.

Reading:  gain = (deviator return under best response) / (its return under the Nash profile) - 1.
  gain ~ 0 for both robots            -> the low-throughput outcome is a Nash equilibrium of the real
                                         game (payoff-dominated by the zone partition of QMIX/VDN).
  gain >> 0 and it comes from tasks  -> not an equilibrium: Nash Q-learning failed to learn the task
                                         (go to test 2: longer training).

Usage (repository root):
  JOBS=12 python3 revision_r16/r16_s3_deviation.py | tee revision_r16/s3_deviation_lab.txt
  env: BR_EPISODES (default 3000), ALGOS (default "nash"; "nash nash_util" for both), SEEDS (default 0-9)
"""
import os, sys, json, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import r16_scalability as R  # noqa: E402

SCN = "S3"
RES = HERE / "results_lab_20261016_115445" / SCN
BR_EPISODES = int(os.environ.get("BR_EPISODES", 3000))
ALGOS = os.environ.get("ALGOS", "nash").split()
SEEDS = [int(x) for x in os.environ.get("SEEDS", " ".join(map(str, range(10)))).split()]
EVAL, EPS = R.EVAL_EPISODES, R.EPS_EVAL
ALPHA = 0.1


def load(algo, seed, n_states):
    d = np.load(RES / algo / f"seed_{seed}_qtables.npz")
    q = [d["q0"], d["q1"]]
    P = np.zeros((2, n_states, 5))
    for s in range(n_states):
        s0, s1, _, _ = R.joint_sel(algo, q[0][s], q[1][s])
        P[0, s], P[1, s] = s0, s1
    P = np.clip(P, 0, None); P /= P.sum(-1, keepdims=True)
    touched = float(np.mean(np.any(q[0].reshape(n_states, -1) != 0, axis=1) |
                            np.any(q[1].reshape(n_states, -1) != 0, axis=1)))
    return q, P, touched


def evaluate(cfg, seed, pol):
    """pol[i](s, rng) -> action; eps = EPS_EVAL for both robots, as in the paper's frozen evaluation."""
    env = R.make_env(SCN, cfg, seed); rng = np.random.default_rng(seed)
    Rs = np.zeros(2); T = np.zeros(2); C = 0
    for _ in range(EVAL):
        s, _ = env.reset()
        for _ in range(env.max_steps):
            a = []
            for i in (0, 1):
                a.append(int(rng.integers(5)) if rng.random() < EPS else pol[i](s, rng))
            s, _, r, _, info = env.step(tuple(a))
            Rs += r; C += info["collisions"]
            for _k, i in info["completed_by"]:
                T[i] += 1
    return {"reward_per_ep": (Rs / EVAL).round(1).tolist(), "tasks_per_ep": (T / EVAL).round(2).tolist(),
            "collisions_per_ep": round(C / EVAL, 2)}


def job(args):
    algo, seed, dev, init = args
    t0 = time.time()
    cfg, _ = R.make_cfg(SCN)
    gamma = float(cfg["learning"].get("gamma", 0.95))
    env = R.make_env(SCN, cfg, seed + 1000)          # training env: different job stream than evaluation
    q, P, touched = load(algo, seed, env.n_states)
    oth = 1 - dev
    rng = np.random.default_rng(10_000 + 100 * seed + 10 * dev + (init == "warm"))
    if init == "warm":
        Q = (np.einsum("sab,sb->sa", q[0], P[1]) if dev == 0 else np.einsum("sab,sa->sb", q[1], P[0])).copy()
    else:
        Q = rng.normal(0, 1e-6, (env.n_states, 5))
    eps, dec = 1.0, 0.05 ** (1.0 / (0.6 * BR_EPISODES))
    for _ in range(BR_EPISODES):
        s, _ = env.reset(); done = False
        while not done:
            a = [0, 0]
            a[oth] = int(rng.integers(5)) if rng.random() < EPS else int(rng.choice(5, p=P[oth, s]))
            a[dev] = int(rng.integers(5)) if rng.random() < eps else int(np.argmax(Q[s]))
            ns, _, r, done, _ = env.step(tuple(a))
            Q[s, a[dev]] += ALPHA * (r[dev] + gamma * Q[ns].max() - Q[s, a[dev]])   # bootstraps like Nash learn()
            s = ns
        eps = max(0.05, eps * dec)
    nash = lambda i: (lambda s, g: int(g.choice(5, p=P[i, s])))
    pol = [nash(0), nash(1)]
    base = evaluate(cfg, seed, pol)
    pol[dev] = lambda s, g: int(np.argmax(Q[s]))
    devr = evaluate(cfg, seed, pol)
    b, d = base["reward_per_ep"][dev], devr["reward_per_ep"][dev]
    return {"algo": algo, "seed": seed, "deviator": dev, "init": init, "br_episodes": BR_EPISODES,
            "states_touched_by_nash": round(touched, 3), "nash_profile": base, "deviation": devr,
            "gain": round(d / b - 1, 4) if b > 0 else None, "elapsed_s": round(time.time() - t0, 1)}


def main():
    out = HERE / f"s3_deviation_{time.strftime('%Y%m%d_%H%M%S')}"; out.mkdir()
    jobs = [(a, s, dv, ini) for a in ALGOS for s in SEEDS for dv in (0, 1) for ini in ("warm", "cold")]
    print(f"{len(jobs)} jobs, BR_EPISODES={BR_EPISODES}, eval {EVAL} eps, eps={EPS} -> {out}", flush=True)
    with Pool(int(os.environ.get("JOBS", 12))) as pool:
        res = []
        for r in pool.imap_unordered(job, jobs):
            res.append(r)
            (out / f"{r['algo']}_seed{r['seed']}_dev{r['deviator']}_{r['init']}.json").write_text(json.dumps(r, indent=1))
            print(f"  {r['algo']:9s} seed {r['seed']} dev R{r['deviator']} {r['init']:4s} gain {r['gain']:+.3f} "
                  f"| {r['elapsed_s']:.0f}s", flush=True)
    print("\nalgo      seed dev  touched  R_dev Nash  R_dev BR  gain   tasks dev Nash->BR  tasks other  R_other Nash->BR")
    best = {}
    for r in res:
        k = (r["algo"], r["seed"], r["deviator"])
        if k not in best or (r["gain"] or -9) > (best[k]["gain"] or -9):
            best[k] = r
    for k in sorted(best):
        r = best[k]; dv, ot = r["deviator"], 1 - r["deviator"]; nb, bd = r["nash_profile"], r["deviation"]
        print(f"{k[0]:9s} {k[1]:4d}  R{dv}   {r['states_touched_by_nash']:.2f}   {nb['reward_per_ep'][dv]:8.1f}  "
              f"{bd['reward_per_ep'][dv]:8.1f}  {r['gain']:+.3f}  {nb['tasks_per_ep'][dv]:6.2f} -> {bd['tasks_per_ep'][dv]:6.2f}"
              f"   {nb['tasks_per_ep'][ot]:5.2f}->{bd['tasks_per_ep'][ot]:5.2f}   "
              f"{nb['reward_per_ep'][ot]:7.1f} -> {bd['reward_per_ep'][ot]:7.1f}  ({r['init']})")
    for a in ALGOS:
        g = np.array([r["gain"] for k, r in best.items() if k[0] == a and r["gain"] is not None])
        print(f"\n{a}: best-response gain median {np.median(g):+.3f}, min {g.min():+.3f}, max {g.max():+.3f}; "
              f"{int((g > 0.05).sum())}/{len(g)} robot-seeds gain > 5 %")
    print("DONE ->", out)


if __name__ == "__main__":
    main()
