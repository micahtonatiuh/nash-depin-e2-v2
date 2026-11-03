#!/usr/bin/env python3
"""
r16_scalability.py — Reviewer 1, point 6 (scalability). Paper 1, Access-2026-41887.

Trains Nash-Q (egalitarian, as published; and utilitarian selection, as in the R1.2 ablation),
QMIX and VDN, unchanged, on larger / richer versions of the
E2 task and evaluates the frozen policies with the R1.3 metrics. Original files are only
imported, never modified.

Scenarios (2 robots, E2 rewards: +task, -5 collision, +2 cooperation at Manhattan distance 1,
200 steps/episode, starts R0=(0,0) on station A and R1=(g,g)):
  E2ref : 5x5, one station A=(0,0) +10                            (control: must equal E2)
  S1    : 7x7, one station A=(0,0) +10                            (larger arena)
  S2    : 5x5, two simultaneous stations A=(0,0) +10, B=(0,4) +5  (as many tasks as robots)
  S3    : 5x5 factory, shelf (2,1)-(2,3); ONE job at a time (+10) that appears at a random
          station among (0,0), (0,4), (4,0) after each completion   (dynamic job arrivals)
Fixed stations respawn in place every step (as the single E2 task does). In S3 the active
station is part of the state (Nash) and of the observation/global state (QMIX/VDN: 6-D).
Obstacles revert a move before the collision check.

Training follows experiments/run_e2.py (same agents, hyperparameters, loop). Episodes scale
with the state space: E2ref/S2 1000 (625 states), S3 2000 (1875), S1 3000 (2401); epsilon decay
is rescaled so that epsilon reaches its 0.05 floor at the same fraction of training.

Evaluation (same rules as R1.3): frozen policy, 100 episodes, eps = 0.05;
Nash samples sigma_i (egalitarian), QMIX/VDN use argmax of each agent network.

Run from the root of nash-depin-e2-v2 on the lab PC:
  python3 revision_r16/r16_scalability.py --check                 # control: E2ref == E2 (~1 min)
  python3 revision_r16/r16_scalability.py --scenario S1 --algo nash --seeds 0 1 2 ...
  bash revision_r16/run_r16_lab.sh                                # everything, in parallel
"""
import argparse, copy, hashlib, json, platform, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np, yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nash_depin.environment_mixed import MixedMotiveGridWorld, Task
from nash_depin.nash_solver import joint_nash, _best_response_iterate, _select
from nash_depin.nash_agent_e2 import NashTeamE2


def joint_util(A, B):
    """Utilitarian selection over the SAME 4 best-response starts as the egalitarian solver
    (as in the R1.2 ablation): max V0 + V1 instead of min |V0 - V1|."""
    n = A.shape[0]; unif = np.ones(n) / n
    e0 = np.zeros(n); e0[0] = 1.0
    e1 = np.zeros(n); e1[min(1, n - 1)] = 1.0
    return _select([_best_response_iterate(A, B, a, b) for a, b in [(unif, unif), (e0, unif), (unif, e0), (e0, e1)]], False)


def joint_sel(algo, A, B):
    return joint_util(A, B) if algo == "nash_util" else joint_nash(A, B, fast=True, equity=True)


class NashTeamUtil(NashTeamE2):
    """Nash-Q identical to NashTeamE2 except for the equilibrium-selection rule (utilitarian)."""

    def learn(self, state, actions, rewards, next_state, obs=None, next_obs=None, done=False):
        q0, q1 = self.agents[0].q_table, self.agents[1].q_table
        s0n, s1n, V0, V1 = joint_util(q0[next_state], q1[next_state])
        nash_next = [V0, V1]
        for i, agent in enumerate(self.agents):
            q = agent.q_table[state, actions[0], actions[1]]
            target = rewards[i] + agent.gamma * nash_next[i]
            agent.q_table[state, actions[0], actions[1]] += agent.alpha * (target - q)
        self.agents[0].policy[next_state] = s0n
        self.agents[1].policy[next_state] = s1n
        s0, s1, _, _ = joint_util(q0[state], q1[state])
        self.agents[0].policy[state] = s0
        self.agents[1].policy[state] = s1

SCENARIOS = {
    "E2ref": dict(grid=5, stations=[((0, 0), 10.0)], obstacles=[], mode="fixed", episodes=1000),
    "S1":    dict(grid=7, stations=[((0, 0), 10.0)], obstacles=[], mode="fixed", episodes=3000),
    "S2":    dict(grid=5, stations=[((0, 0), 10.0), ((0, 4), 5.0)], obstacles=[], mode="fixed", episodes=1000),
    "S3":    dict(grid=5, stations=[((0, 0), 10.0), ((0, 4), 10.0), ((4, 0), 10.0)],
                  obstacles=[(2, 1), (2, 2), (2, 3)], mode="dynamic", episodes=2000),
}
EPS_EVAL, EVAL_EPISODES = 0.05, 100


class ScenarioGridWorld(MixedMotiveGridWorld):
    """E2 environment with stations of individual value, static obstacles and, in 'dynamic'
    mode, a single job that moves to a random station after each completion.
    Step logic = MixedMotiveGridWorld.step plus (a) moves into an obstacle are reverted before
    the collision check, (b) fixed stations respawn in place, (c) dynamic job relocation."""

    def __init__(self, grid, stations, obstacles, mode, cfg_env, seed=None):
        self.stations, self.obstacles, self.mode = stations, set(obstacles), mode
        self.K = len(stations)
        self.job = 0
        super().__init__(grid_size=grid, n_agents=2, n_tasks=1 if mode == "dynamic" else self.K,
                         task_reward=cfg_env["task_reward"], collision_penalty=cfg_env["collision_penalty"],
                         cooperation_reward=cfg_env["cooperation_reward"], coop_distance=cfg_env["coop_distance"],
                         max_steps=cfg_env["max_steps"], seed=seed)
        if mode == "dynamic":
            self.n_states = grid ** 4 * self.K
            self.OBS_DIM = 6

    def _setup_task_positions(self):
        self._task_positions = [p for p, _ in self.stations]

    def _active(self):
        if self.mode == "dynamic":
            return [(self.job, *self.stations[self.job])]
        return [(k, p, r) for k, (p, r) in enumerate(self.stations)]

    def reset(self):
        super().reset()
        self.tasks = []
        if self.mode == "dynamic":
            self.job = int(self.rng.integers(self.K))
        return self._encode_state(), self._get_obs()

    def _encode_state(self):
        s = 0
        for r, c in self.agent_positions:
            s = s * self.grid_size * self.grid_size + r * self.grid_size + c
        return s * self.K + self.job if self.mode == "dynamic" else s % self.n_states

    def _get_obs(self):
        if self.mode != "dynamic":
            return super()._get_obs()
        g = self.grid_size - 1; jr, jc = self.stations[self.job][0]
        (r0, c0), (r1, c1) = self.agent_positions
        obs = np.array([[r0 / g, c0 / g, r1 / g, c1 / g], [r1 / g, c1 / g, r0 / g, c0 / g]], dtype=np.float32)
        return np.hstack([obs, np.tile([jr / g, jc / g], (2, 1))]).astype(np.float32)

    @property
    def state_dim(self):
        return 6 if self.mode == "dynamic" else 4

    def get_global_state_vec(self):
        v = super().get_global_state_vec()
        if self.mode != "dynamic":
            return v
        g = self.grid_size - 1; jr, jc = self.stations[self.job][0]
        return np.append(v, [jr / g, jc / g]).astype(np.float32)

    def step(self, actions):
        self.step_count += 1
        rewards = [0.0] * self.n_agents
        info = {"collisions": 0, "tasks_completed": 0, "cooperations": 0, "completed_by": []}
        new_positions = []
        for i, action in enumerate(actions):
            dr, dc = self.ACTIONS[action]
            r, c = self.agent_positions[i]
            nr = max(0, min(self.grid_size - 1, r + dr)); nc = max(0, min(self.grid_size - 1, c + dc))
            if (nr, nc) in self.obstacles:
                nr, nc = r, c
            new_positions.append((nr, nc))
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if new_positions[i] == new_positions[j]:
                    rewards[i] += self.collision_penalty; rewards[j] += self.collision_penalty
                    info["collisions"] += 1
                    new_positions[i] = self.agent_positions[i]; new_positions[j] = self.agent_positions[j]
        self.agent_positions = new_positions
        done_job = False
        for k, pos_t, val in self._active():
            for i, pos in enumerate(self.agent_positions):
                if pos == pos_t:
                    rewards[i] += val; info["tasks_completed"] += 1
                    info["completed_by"].append((k, i)); done_job = True
                    break
        if self.n_agents == 2 and info["collisions"] == 0:
            (r0, c0), (r1, c1) = self.agent_positions
            if 0 < abs(r0 - r1) + abs(c0 - c1) <= self.coop_distance:
                rewards[0] += self.cooperation_reward; rewards[1] += self.cooperation_reward
                info["cooperations"] += 1
        if self.mode == "dynamic" and done_job:
            self.job = int(self.rng.integers(self.K))
        done = self.step_count >= self.max_steps
        return self._encode_state(), self._get_obs(), rewards, done, info


def make_cfg(scn):
    cfg = yaml.safe_load(open(ROOT / "config" / "e2_config.yaml"))
    episodes = SCENARIOS[scn]["episodes"]
    cfg["learning"]["epsilon_decay"] = 0.995 ** (1000 / episodes)
    return cfg, episodes


def make_env(scn, cfg, seed):
    s = SCENARIOS[scn]
    return ScenarioGridWorld(s["grid"], s["stations"], s["obstacles"], s["mode"], cfg["environment"], seed=seed)


def build_team(algo, env, cfg, seed):
    if algo == "nash":
        return NashTeamE2(env.n_states, env.N_ACTIONS, cfg, seed=seed)
    if algo == "nash_util":
        return NashTeamUtil(env.n_states, env.N_ACTIONS, cfg, seed=seed)
    if algo == "vdn":
        from baselines.vdn_baseline import VDNTeam
        return VDNTeam(env.OBS_DIM, env.N_ACTIONS, cfg, seed=seed)
    if algo == "qmix":
        from baselines.qmix_baseline import QMIXTeam
        return QMIXTeam(env.OBS_DIM, env.state_dim, env.N_ACTIONS, cfg, seed=seed)
    raise ValueError(algo)


def train(algo, scn, seed, episodes=None):
    cfg, n_ep = make_cfg(scn)
    episodes = episodes or n_ep
    env = make_env(scn, cfg, seed); team = build_team(algo, env, cfg, seed)
    hist = []
    for ep in range(episodes):
        state, obs = env.reset(); sv = env.get_global_state_vec()
        R = np.zeros(2); T = np.zeros((2, len(env.stations))); C = 0; done = False
        while not done:
            a = team.act(state, obs)
            ns, nobs, r, done, info = env.step(a); nsv = env.get_global_state_vec()
            if algo.startswith("nash"):
                team.learn(state, a, r, ns)
            elif algo == "vdn":
                team.learn(state, a, r, ns, obs, nobs, done)
            else:
                team.learn(sv, a, r, nsv, obs, nobs, done)
            R += r; C += info["collisions"]
            for k, i in info["completed_by"]: T[i, k] += 1
            state, obs, sv = ns, nobs, nsv
        team.end_episode(); hist.append((R.tolist(), T.tolist(), C))
    return env, team, hist, episodes


def policy_fn(algo, team, env):
    if algo.startswith("nash"):
        q0, q1 = team.agents[0].q_table, team.agents[1].q_table
        P = [np.zeros((env.n_states, 5)), np.zeros((env.n_states, 5))]
        for s in range(env.n_states):
            s0, s1, _, _ = joint_sel(algo, q0[s], q1[s])
            P[0][s], P[1][s] = s0, s1
        def act(i, s, obs, rng):
            p = np.clip(P[i][s], 0, None); return int(rng.choice(5, p=p / p.sum()))
        return act
    import torch
    def act(i, s, obs, rng):
        with torch.no_grad():
            return int(team.q_nets[i](torch.tensor(obs[i], dtype=torch.float32)).argmax().item())
    return act


def gini2(x):
    x = np.asarray(x, float); return float(abs(x[0] - x[1]) / (2 * x.sum())) if x.sum() > 0 else float("nan")


def evaluate(algo, team, scn, seed):
    cfg, _ = make_cfg(scn); env = make_env(scn, cfg, seed); act = policy_fn(algo, team, env)
    rng = np.random.default_rng(seed); K = len(env.stations); eps = []
    key = (lambda k: k) if env.mode == "fixed" else (lambda k: 0)
    for _ in range(EVAL_EPISODES):
        s, obs = env.reset(); R = np.zeros(2); T = np.zeros((2, K)); C = 0; mv = 0; ho = 0; last = [None] * K
        for _ in range(env.max_steps):
            a = []
            for i in (0, 1):
                a.append(int(rng.integers(5)) if rng.random() < EPS_EVAL else act(i, s, obs, rng))
            before = list(env.agent_positions)
            s, obs, r, _, info = env.step(tuple(a))
            R += r; C += info["collisions"]; mv += sum(before[i] != env.agent_positions[i] for i in (0, 1))
            for k, i in info["completed_by"]:
                T[i, k] += 1
                if last[key(k)] is not None and last[key(k)] != i: ho += 1
                last[key(k)] = i
        eps.append((R, T, C, mv, ho))
    Rs = np.sum([e[0] for e in eps], axis=0); Ts = np.sum([e[1] for e in eps], axis=0); n = len(eps)
    A = Ts[:, 0]
    return {"gini": round(gini2(Rs), 4),
            "ep_gini_median": round(float(np.nanmedian([gini2(e[0]) for e in eps])), 4),
            "reward_per_agent_per_ep": (Rs / n).round(1).tolist(),
            "total_reward_per_ep": round(float(Rs.sum() / n), 1),
            "tasks_per_ep_by_station": (Ts / n).round(2).tolist(),
            "tasks_per_ep": round(float(Ts.sum() / n), 2),
            "top_task_share": round(float(Ts.sum(1).max() / Ts.sum()), 4) if Ts.sum() > 0 else None,
            "station_A_top_share": round(float(A.max() / A.sum()), 4) if A.sum() > 0 else None,
            "handovers_per_ep": round(float(np.mean([e[4] for e in eps])), 2),
            "episodes_with_handover": int(sum(e[4] > 0 for e in eps)),
            "moves_per_ep": round(float(np.mean([e[3] for e in eps])), 1),
            "collisions_per_ep": round(float(np.mean([e[2] for e in eps])), 2)}


def run(algo, scn, seed, out, episodes=None):
    t0 = time.time()
    env, team, hist, n_ep = train(algo, scn, seed, episodes)
    last = hist[-100:]
    R = np.mean([h[0] for h in last], axis=0)
    res = {"algo": algo, "scenario": scn, "seed": seed, "episodes": n_ep,
           "epsilon_decay": make_cfg(scn)[0]["learning"]["epsilon_decay"],
           "train_last100": {"gini": round(gini2(R), 4), "reward_per_agent": R.round(1).tolist(),
                             "tasks_by_station": np.mean([h[1] for h in last], axis=0).round(2).tolist(),
                             "collisions": round(float(np.mean([h[2] for h in last])), 2)},
           "frozen": evaluate(algo, team, scn, seed), "elapsed_s": round(time.time() - t0, 1)}
    d = out / scn / algo; d.mkdir(parents=True, exist_ok=True)
    (d / f"seed_{seed}.json").write_text(json.dumps(res, indent=1))
    if algo.startswith("nash"):
        np.savez_compressed(d / f"seed_{seed}_qtables.npz", q0=team.agents[0].q_table, q1=team.agents[1].q_table)
    else:
        import torch
        torch.save({f"agent_{i}": team.q_nets[i].state_dict() for i in (0, 1)}, d / f"seed_{seed}_agents.pt")
    f = res["frozen"]
    print(f"{scn:5s} {algo:9s} seed {seed:2d}: Gini {f['gini']:.4f} epGini {f['ep_gini_median']:.3f} "
          f"tasks {f['tasks_per_ep_by_station']} top {f['top_task_share']} A-share {f['station_A_top_share']} ho {f['handovers_per_ep']:.2f} "
          f"coll {f['collisions_per_ep']:.2f} R {f['total_reward_per_ep']:.0f} | {res['elapsed_s']:.0f}s", flush=True)
    return res


def check():
    """Control: the scenario environment with E2ref must reproduce the E2 environment exactly."""
    cfg, _ = make_cfg("E2ref"); ok = True
    for seed in range(3):
        base = MixedMotiveGridWorld(grid_size=5, n_agents=2, n_tasks=1, max_steps=200, seed=seed)
        scn = make_env("E2ref", cfg, seed); rng = np.random.default_rng(100 + seed)
        for _ in range(5):
            b = base.reset(); s = scn.reset(); ok &= b[0] == s[0]
            for _ in range(200):
                a = (int(rng.integers(5)), int(rng.integers(5)))
                rb = base.step(a); rs = scn.step(a)
                ok &= rb[0] == rs[0] and list(rb[2]) == list(rs[2]) and rb[3] == rs[3] and rb[4]["collisions"] == rs[4]["collisions"]
    print("env control (E2ref vs MixedMotiveGridWorld, 3 seeds x 5 eps x 200 random steps):", "PASS" if ok else "FAIL")
    # Nash training control: seed 0 with E2ref must give the published Q-tables (lab PC only: bit-exact BLAS path)
    _, team, _, _ = train("nash", "E2ref", 0)
    pub = [np.load(ROOT / "qtables_all_seeds" / f"seed_0_agent_{i}.npz")["q_table"] for i in (0, 1)]
    same = all(np.array_equal(team.agents[i].q_table, pub[i]) for i in (0, 1))
    print("Nash training control (E2ref seed 0 vs qtables_all_seeds/seed_0):", "PASS (bit-exact)" if same else
          f"DIFFERENT (max |dQ| = {max(np.abs(team.agents[i].q_table - pub[i]).max() for i in (0, 1)):.3g}; expected only off the lab PC)")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--scenario", choices=list(SCENARIOS))
    ap.add_argument("--algo", choices=["nash", "nash_util", "qmix", "vdn"])
    ap.add_argument("--seeds", type=int, nargs="*", default=list(range(10)))
    ap.add_argument("--episodes", type=int, default=None, help="override (timing tests only)")
    ap.add_argument("--out", default=str(ROOT / "revision_r16" / "results"))
    a = ap.parse_args()
    if a.check:
        sys.exit(0 if check() else 1)
    out = Path(a.out)
    for sd in a.seeds:
        run(a.algo, a.scenario, sd, out, a.episodes)
