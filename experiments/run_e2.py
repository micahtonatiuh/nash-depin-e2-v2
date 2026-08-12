"""
run_e2.py — Main orchestrator for Experiment E2
Nash-DePIN: Nash vs QMIX/VDN (fairness analysis).

Runs ONE algorithm over N seeds and logs fairness metrics (Gini, Jain, task
balance) averaged over the final episodes of each run.

Usage:
    python3 experiments/run_e2.py --algo nash --seeds 10 --output e2_results/nash/
    python3 experiments/run_e2.py --algo qmix --seeds 10 --output e2_results/qmix/
    python3 experiments/run_e2.py --algo vdn  --seeds 10 --output e2_results/vdn/

    # Quick check (2 seeds, 50 episodes):
    python3 experiments/run_e2.py --algo nash --seeds 2 --output e2_results/nash/ --dry_run

Note: the descriptive summary written here (mean +/- std) is superseded for
the paper by experiments/analyze_e2_stats_v2.py, which reports the MEDIAN and
non-parametric tests because the Nash Gini distribution is bimodal.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from nash_depin.environment_mixed import MixedMotiveGridWorld
from nash_depin.fairness_metrics import compute_all_fairness
from nash_depin.nash_agent_e2 import NashTeamE2
from baselines.vdn_baseline import VDNTeam
from baselines.qmix_baseline import QMIXTeam


def setup_logging(output_dir):
    log_path = Path(output_dir) / "experiment.log"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if root.handlers:
        root.handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )


def build_team(algo, env, cfg, seed):
    """Builds the agent team for the chosen algorithm."""
    if algo == "nash":
        return NashTeamE2(env.n_states, env.N_ACTIONS, cfg, seed=seed)
    elif algo == "vdn":
        return VDNTeam(env.OBS_DIM, env.N_ACTIONS, cfg, seed=seed)
    elif algo == "qmix":
        return QMIXTeam(env.OBS_DIM, env.state_dim, env.N_ACTIONS, cfg, seed=seed)
    raise ValueError(f"Unknown algorithm: {algo}")


def run_one_seed(algo, cfg, seed, episodes, log):
    """
    Trains ONE algorithm with ONE seed. Returns fairness metrics averaged
    over the last `eval_last_n` episodes.
    """
    env_cfg = cfg["environment"]
    env = MixedMotiveGridWorld(
        grid_size=env_cfg["grid_size"], n_agents=env_cfg["n_agents"],
        n_tasks=env_cfg["num_tasks"], task_reward=env_cfg["task_reward"],
        collision_penalty=env_cfg["collision_penalty"],
        cooperation_reward=env_cfg["cooperation_reward"],
        coop_distance=env_cfg["coop_distance"], max_steps=env_cfg["max_steps"],
        seed=seed,
    )
    team = build_team(algo, env, cfg, seed)
    eval_last_n = cfg["experiment_control"]["eval_last_n"]

    # Per-episode history of rewards and completed tasks, per agent
    ep_rewards = []   # [(r0, r1), ...]
    ep_tasks = []     # [(t0, t1), ...]
    ep_collisions = []

    for episode in range(1, episodes + 1):
        state, obs = env.reset()
        state_vec = env.get_global_state_vec()
        cum_rewards = [0.0, 0.0]
        cum_tasks = [0, 0]
        cum_collisions = 0

        done = False
        while not done:
            actions = team.act(state, obs)
            next_state, next_obs, rewards, done, info = env.step(actions)
            next_state_vec = env.get_global_state_vec()

            # Train (uniform call signature across algorithms)
            if algo == "nash":
                team.learn(state, actions, rewards, next_state)
            elif algo == "vdn":
                team.learn(state, actions, rewards, next_state, obs, next_obs, done)
            else:  # qmix
                team.learn(state_vec, actions, rewards, next_state_vec,
                           obs, next_obs, done)

            # Attribute each completed task to the agent whose reward
            for i in range(2):
                cum_rewards[i] += rewards[i]
            cum_tasks_step = info["tasks_completed"]
            # increased by at least +task_reward on this step
            if cum_tasks_step > 0:
                for i in range(2):
                    if rewards[i] >= env_cfg["task_reward"]:
                        cum_tasks[i] += 1
            cum_collisions += info["collisions"]

            state, obs, state_vec = next_state, next_obs, next_state_vec

        team.end_episode()
        ep_rewards.append(tuple(cum_rewards))
        ep_tasks.append(tuple(cum_tasks))
        ep_collisions.append(cum_collisions)

        if episode % 200 == 0:
            log.info(f"  [{algo} seed {seed}] ep {episode}/{episodes} | "
                     f"ε={getattr(team,'epsilon',0):.3f} | "
                     f"R=({cum_rewards[0]:.0f},{cum_rewards[1]:.0f})")

    # Average rewards and tasks over the last N episodes
    last_rewards = np.array(ep_rewards[-eval_last_n:])
    last_tasks = np.array(ep_tasks[-eval_last_n:])
    mean_rewards = last_rewards.mean(axis=0)        # [mean_r0, mean_r1]
    mean_tasks = last_tasks.mean(axis=0)            # [mean_t0, mean_t1]

    fairness = compute_all_fairness(mean_rewards.tolist(), mean_tasks.tolist())
    fairness["mean_collisions"] = float(np.mean(ep_collisions[-eval_last_n:]))
    fairness["seed"] = seed
    return fairness


def main():
    parser = argparse.ArgumentParser(description="E2 — Nash vs QMIX/VDN Fairness")
    parser.add_argument("--config", default="config/e2_config.yaml")
    parser.add_argument("--algo", required=True, choices=["nash", "qmix", "vdn"])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry_run", action="store_true",
                        help="2 seeds, 50 episodes — setup check only")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    setup_logging(args.output)
    log = logging.getLogger(__name__)

    episodes = cfg["experiment_control"]["episodes"]
    n_seeds = args.seeds
    if args.dry_run:
        episodes, n_seeds = 50, 2
        log.info("=== DRY RUN — 2 seeds, 50 episodes ===")

    log.info("=" * 60)
    log.info(f"EXPERIMENT E2 — {args.algo.upper()} | {n_seeds} seeds")
    log.info("=" * 60)

    env_cfg = cfg["environment"]
    log.info(f"Environment: {env_cfg['grid_size']}x{env_cfg['grid_size']} mixed-motive | "
             f"+{env_cfg['task_reward']}/task {env_cfg['collision_penalty']}/collision "
             f"+{env_cfg['cooperation_reward']}/coop")
    log.info(f"Episodes per seed: {episodes}")

    start = time.time()
    results = []
    for seed in range(n_seeds):
        log.info(f"\n--- {args.algo} | seed {seed+1}/{n_seeds} ---")
        t0 = time.time()
        res = run_one_seed(args.algo, cfg, seed, episodes, log)
        results.append(res)
        log.info(f"  Gini={res['gini']:.4f} | Jain={res['jain']:.4f} | "
                 f"Balance={res.get('task_balance',0):.3f} | "
                 f"({time.time()-t0:.0f}s)")

    # Aggregate summary
    ginis = [r["gini"] for r in results]
    jains = [r["jain"] for r in results]
    totals = [r["total_reward"] for r in results]
    balances = [r.get("task_balance", 0) for r in results]

    summary = {
        "algo": args.algo,
        "n_seeds": n_seeds,
        "episodes": episodes,
        "gini_mean": float(np.mean(ginis)), "gini_std": float(np.std(ginis)),
        "jain_mean": float(np.mean(jains)), "jain_std": float(np.std(jains)),
        "total_reward_mean": float(np.mean(totals)), "total_reward_std": float(np.std(totals)),
        "task_balance_mean": float(np.mean(balances)),
        "per_seed": results,
        "elapsed_min": round((time.time() - start) / 60, 1),
    }

    out = Path(args.output)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("\n" + "=" * 60)
    log.info(f"SUMMARY {args.algo.upper()} ({n_seeds} seeds)")
    log.info(f"  Gini:  {summary['gini_mean']:.4f} +/- {summary['gini_std']:.4f}")
    log.info(f"  Jain:  {summary['jain_mean']:.4f} +/- {summary['jain_std']:.4f}")
    log.info(f"  Total reward: {summary['total_reward_mean']:.1f} +/- {summary['total_reward_std']:.1f}")
    log.info(f"  Task balance: {summary['task_balance_mean']:.3f}")
    log.info(f"  Elapsed: {summary['elapsed_min']} min")
    log.info("=" * 60)

    # Quick check against the preregistered hypothesis
    hyp = cfg["hypothesis"]
    if args.algo == "nash":
        ok = summary["gini_mean"] < hyp["nash_gini_max"]
        log.info(f"\n{'PASS' if ok else 'FAIL'}: Nash Gini {summary['gini_mean']:.4f} "
                 f"{'<' if ok else '>='} {hyp['nash_gini_max']} (target)")
    else:
        ok = summary["gini_mean"] > hyp["baseline_gini_min"]
        log.info(f"\n{'PASS' if ok else 'FAIL'}: {args.algo.upper()} Gini {summary['gini_mean']:.4f} "
                 f"{'>' if ok else '<='} {hyp['baseline_gini_min']} (expected)")

    log.info("Once all 3 algorithms are done, run: experiments/analyze_e2_stats_v2.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
