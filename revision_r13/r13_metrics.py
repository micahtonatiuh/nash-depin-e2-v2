#!/usr/bin/env python3
"""
r13_metrics.py — Reviewer 1, point 3 (metrics beyond Gini). Paper 1, Access-2026-41887.

Re-evaluates the FROZEN policies trained for R1.2 (no learning, no file is modified) in the
E2 environment and E2 start (R0=(0,0), R1=(4,4)), 100 episodes x 200 steps per seed, eps = 0.05,
and logs per-episode quantities that the R1.2 runs did not store:

  per-agent reward, total reward, pooled and per-episode reward Gini, Jain index,
  tasks per robot, top task share, hand-overs (completer differs from the previous completer),
  moves (cells actually changed), collisions.

Policies (same behaviour rules as the R1.2 frozen evaluation):
  Nash-Q egalitarian / single / utilitarian : sample sigma_i w.p. 1-eps, uniform w.p. eps
      egalitarian = joint_nash(fast, equity=True)       (4 best-response starts, min |V0-V1|)
      single      = joint_nash(fast, equity=False)      (1 start, code flag USE_EQUITY=False)
      utilitarian = same 4 starts as egalitarian, max V0+V1
  QMIX / VDN / QPLEX : argmax of each agent network w.p. 1-eps, uniform w.p. eps
      (decentralized execution; the mixer is not used at execution time)

Control: the pooled Gini, tasks and collisions are compared with the values stored by R1.2
(sample_eps for Nash-Q, argmax_eps for the MARL baselines). Same RNG stream -> identical values
expected; any difference is reported, not hidden.

Run from the root of nash-depin-e2-v2 on the lab PC:
    python3 revision_r13/r13_metrics.py            # all methods, seeds 0-19
    python3 revision_r13/r13_metrics.py --quick    # seed 0 of each method (~1-2 min)
Outputs: revision_r13/results_r13_<timestamp>/ (per_seed.json, summary.txt, env.txt, SHA256SUMS)
"""
import argparse, hashlib, json, platform, subprocess, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nash_depin.environment_mixed import MixedMotiveGridWorld
from nash_depin.nash_solver import joint_nash, _best_response_iterate, _select

R12 = ROOT / "revision_r12"
NASH_DIR = R12 / "results_lab_20261014_201553"
MARL_DIR = {"qmix": R12 / "results_qmix_lab_rerun" / "qmix",
            "vdn": R12 / "results_vdn_lab_rerun" / "vdn",
            "qplex": R12 / "results_qplex_lab_20261015_072008" / "qplex"}
START = [(0, 0), (4, 4)]
TASK = (0, 0)
EPS, EPISODES, STEPS = 0.05, 100, 200


# ---------------------------------------------------------------- policies
def nash_policy(variant, seed):
    q = [np.load(NASH_DIR / variant / "qtables" / f"seed_{seed}_agent_{i}.npz")["q_table"] for i in (0, 1)]
    P = [np.zeros((625, 5)), np.zeros((625, 5))]
    for s in range(625):
        A, B = q[0][s], q[1][s]
        if variant == "egalitarian":
            s0, s1, _, _ = joint_nash(A, B, fast=True, equity=True)
        elif variant == "single":
            s0, s1, _, _ = joint_nash(A, B, fast=True, equity=False)
        elif variant == "utilitarian":
            n = A.shape[0]; unif = np.ones(n) / n
            e0 = np.zeros(n); e0[0] = 1.0
            e1 = np.zeros(n); e1[min(1, n - 1)] = 1.0
            cands = [_best_response_iterate(A, B, a, b) for a, b in [(unif, unif), (e0, unif), (unif, e0), (e0, e1)]]
            s0, s1, _, _ = _select(cands, False)
        else:
            raise ValueError(variant)
        P[0][s], P[1][s] = s0, s1
    return P


def marl_nets(algo, seed):
    import torch
    from baselines.qmix_baseline import AgentNet
    sd = torch.load(MARL_DIR[algo] / f"seed_{seed}_agents.pt", map_location="cpu", weights_only=True)
    nets = []
    for i in (0, 1):
        net = AgentNet(4, 5, sd[f"agent_{i}"]["fc1.weight"].shape[0]); net.load_state_dict(sd[f"agent_{i}"]); net.eval()
        nets.append(net)
    return nets


# ---------------------------------------------------------------- rollout
def rollout(kind, pol, seed):
    rng = np.random.default_rng(seed)
    env = MixedMotiveGridWorld(grid_size=5, n_agents=2, n_tasks=1, max_steps=STEPS, seed=seed)
    if kind == "marl":
        import torch
    ep = []
    for _ in range(EPISODES):
        env.reset(); env.agent_positions = list(START); s = env._encode_state(); obs = env._get_obs()
        er = np.zeros(2); tasks = np.zeros(2); coll = 0; moves = 0; ho = 0; last = None
        for _ in range(STEPS):
            acts = []
            for i in (0, 1):
                if rng.random() < EPS:
                    acts.append(int(rng.integers(5)))
                elif kind == "nash":
                    p = np.clip(pol[i][s], 0, None); acts.append(int(rng.choice(5, p=p / p.sum())))
                else:
                    with torch.no_grad():
                        acts.append(int(pol[i](torch.tensor(obs[i], dtype=torch.float32)).argmax().item()))
            before = list(env.agent_positions)
            s, obs, r, _, info = env.step(tuple(acts))
            er += r; coll += info["collisions"]
            moves += sum(before[i] != env.agent_positions[i] for i in (0, 1))
            for i in (0, 1):
                if env.agent_positions[i] == TASK:          # task fixed at (0,0); same rule as reeval_frozen_e2
                    tasks[i] += 1
                    if last is not None and last != i: ho += 1
                    last = i
        ep.append(dict(r=er.tolist(), tasks=tasks.tolist(), coll=coll, moves=moves, ho=ho))
    return ep


def gini2(x):
    x = np.asarray(x, float); return float(abs(x[0] - x[1]) / (2 * x.sum())) if x.sum() > 0 else float("nan")


def summarize(ep):
    R = np.sum([e["r"] for e in ep], axis=0); T = np.sum([e["tasks"] for e in ep], axis=0)
    n = len(ep); g = gini2(R)
    return {"gini": round(g, 4), "jain": round(float(R.sum() ** 2 / (2 * (R ** 2).sum())), 4),
            "ep_gini_median": round(float(np.nanmedian([gini2(e["r"]) for e in ep])), 4),
            "reward_per_agent_per_ep": (R / n).round(1).tolist(), "total_reward_per_ep": round(float(R.sum() / n), 1),
            "tasks_per_ep": (T / n).round(2).tolist(), "top_task_share": round(float(T.max() / T.sum()), 4),
            "task_mostly_r1": bool(T[1] > T[0]),
            "handovers_per_ep": round(float(np.mean([e["ho"] for e in ep])), 2),
            "episodes_with_handover": int(sum(e["ho"] > 0 for e in ep)),
            "moves_per_ep": round(float(np.mean([e["moves"] for e in ep])), 1),
            "collisions_per_ep": round(float(np.mean([e["coll"] for e in ep])), 2)}


def stored(kind, name, seed):
    if kind == "nash":
        f = json.load(open(NASH_DIR / name / f"seed_{seed}.json"))["frozen_e2"]["sample_eps"]
    else:
        f = json.load(open(MARL_DIR[name] / f"seed_{seed}.json"))["frozen_e2"]["argmax_eps"]
    return f


def control(m, f):
    ok = {"gini": abs(m["gini"] - f["gini"]) < 5e-5,
          "tasks": all(abs(a - b) < 0.051 for a, b in zip(m["tasks_per_ep"], f["tasks_per_ep"])),
          "collisions": abs(m["collisions_per_ep"] - f["collisions_per_ep"]) < 0.0051}
    if "handovers_per_ep" in f: ok["handovers"] = abs(m["handovers_per_ep"] - f["handovers_per_ep"]) < 0.0051
    if "total_reward_per_ep" in f: ok["reward"] = abs(m["total_reward_per_ep"] - f["total_reward_per_ep"]) < 0.051
    return ok


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--methods", nargs="*", default=["egalitarian", "single", "utilitarian", "qmix", "vdn", "qplex"])
    a = ap.parse_args()
    seeds = [0] if a.quick else list(range(20))
    out_dir = ROOT / "revision_r13" / f"results_r13_{datetime.now():%Y%m%d_%H%M%S}"; out_dir.mkdir(parents=True)
    import numpy
    env_lines = [f"host: {platform.node()}  date: {datetime.now().isoformat(timespec='seconds')}",
                 f"python {platform.python_version()} numpy {numpy.__version__}"]
    try:
        import torch; env_lines.append(f"torch {torch.__version__}")
    except ImportError:
        env_lines.append("torch not available")
    for p in ["revision_r13/r13_metrics.py", "nash_depin/environment_mixed.py", "nash_depin/nash_solver.py", "baselines/qmix_baseline.py"]:
        env_lines.append(f"{hashlib.sha256((ROOT / p).read_bytes()).hexdigest()}  {p}")
    (out_dir / "env.txt").write_text("\n".join(env_lines) + "\n"); print("\n".join(env_lines))
    res = {}
    for name in a.methods:
        kind = "nash" if name in ("egalitarian", "single", "utilitarian") else "marl"
        res[name] = []
        for seed in seeds:
            t0 = time.time()
            pol = nash_policy(name, seed) if kind == "nash" else marl_nets(name, seed)
            m = summarize(rollout(kind, pol, seed)); m["seed"] = seed
            m["control_vs_r12"] = control(m, stored(kind, name, seed))
            res[name].append(m)
            print(f"{name:11s} seed {seed:2d}: Gini {m['gini']:.4f} tasks {m['tasks_per_ep']} ho {m['handovers_per_ep']:.2f} "
                  f"moves {m['moves_per_ep']:.1f} coll {m['collisions_per_ep']:.2f} R {m['total_reward_per_ep']:.0f} "
                  f"control {'OK' if all(m['control_vs_r12'].values()) else m['control_vs_r12']} | {time.time()-t0:.0f}s", flush=True)
    (out_dir / "per_seed.json").write_text(json.dumps(res, indent=1))
    med = lambda v: float(np.median(v))
    lines = ["method       n  Gini  fair  top  R1dom  ho/ep  ep_w_ho  moves  coll  reward  epGini  control"]
    for name, rows in res.items():
        allok = sum(all(r["control_vs_r12"].values()) for r in rows)
        lines.append(f"{name:11s} {len(rows):2d} {med([r['gini'] for r in rows]):.3f} {sum(r['gini'] < 0.15 for r in rows):4d} "
                     f"{med([r['top_task_share'] for r in rows]):.2f} {sum(r['task_mostly_r1'] for r in rows):5d} "
                     f"{med([r['handovers_per_ep'] for r in rows]):6.2f} {med([r['episodes_with_handover'] for r in rows]):7.0f} "
                     f"{med([r['moves_per_ep'] for r in rows]):6.1f} {med([r['collisions_per_ep'] for r in rows]):5.2f} "
                     f"{med([r['total_reward_per_ep'] for r in rows]):7.0f} {med([r['ep_gini_median'] for r in rows]):.3f}  {allok}/{len(rows)}")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n"); print("\n" + "\n".join(lines))
    with open(out_dir / "SHA256SUMS", "w") as fh:
        for p in sorted(out_dir.iterdir()):
            if p.name != "SHA256SUMS": fh.write(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n")
    print(f"\nDONE -> {out_dir}")
