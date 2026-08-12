#!/usr/bin/env python3
"""
validate_e2_convergence.py — Convergence validation for E2 Nash.

Retrains Nash with the SAME 10 seeds (0-9) used in run_e2.py, records the Gini
over growing evaluation windows plus a per-episode trajectory, and issues a
per-seed verdict: 'stable' (converged to an equilibrium) vs 'oscillating' (not
converged). Writes to a separate directory and does not modify e2_results/.

Scientific purpose: determine whether the high-Gini seeds are stable monopoly
equilibria (a real finding) or non-converged policies (an artifact).

Usage (from the repository root):
    python3 experiments/validate_e2_convergence.py \
        --config config/e2_config.yaml \
        --seeds 10 \
        --output e2_convergence_validation/

Output:
    e2_convergence_validation/
        validation.log
        seed_{s}_trajectory.csv         # ep, r0, r1, t0, t1, collisions, gini_running
        convergence_summary.json        # per-seed verdict + aggregate
"""
import argparse, json, logging, sys, time
from pathlib import Path
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from nash_depin.environment_mixed import MixedMotiveGridWorld
from nash_depin.fairness_metrics import compute_all_fairness
from nash_depin.nash_agent_e2 import NashTeamE2

WINDOWS = [50, 100, 200]      # final evaluation windows for the stability check
STABILITY_TOL = 0.05          # stable if the Gini spread across windows <= tol


def setup_logging(output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if root.handlers:
        root.handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(Path(output_dir) / "validation.log"),
                  logging.StreamHandler(sys.stdout)],
    )


def gini_over_window(ep_rewards, ep_tasks, last_n):
    """Gini computed exactly as in run_e2.py: rewards/tasks averaged over the last last_n episodes."""
    r = np.array(ep_rewards[-last_n:]).mean(axis=0)
    t = np.array(ep_tasks[-last_n:]).mean(axis=0)
    return compute_all_fairness(r.tolist(), t.tolist())["gini"]


def run_seed(cfg, seed, episodes, out_dir, log):
    env_cfg = cfg["environment"]
    env = MixedMotiveGridWorld(
        grid_size=env_cfg["grid_size"], n_agents=env_cfg["n_agents"],
        n_tasks=env_cfg["num_tasks"], task_reward=env_cfg["task_reward"],
        collision_penalty=env_cfg["collision_penalty"],
        cooperation_reward=env_cfg["cooperation_reward"],
        coop_distance=env_cfg["coop_distance"], max_steps=env_cfg["max_steps"],
        seed=seed,
    )
    team = NashTeamE2(env.n_states, env.N_ACTIONS, cfg, seed=seed)
    task_reward = env_cfg["task_reward"]

    ep_rewards, ep_tasks, ep_collisions = [], [], []
    csv_path = out_dir / f"seed_{seed}_trajectory.csv"
    with open(csv_path, "w") as f:
        f.write("episode,r0,r1,t0,t1,collisions,gini_running100\n")

    for episode in range(1, episodes + 1):
        state, obs = env.reset()
        cum_r = [0.0, 0.0]; cum_t = [0, 0]; cum_c = 0
        done = False
        while not done:
            actions = team.act(state, obs)
            next_state, next_obs, rewards, done, info = env.step(actions)
            team.learn(state, actions, rewards, next_state)
            for i in range(2):
                cum_r[i] += rewards[i]
                if info["tasks_completed"] > 0 and rewards[i] >= task_reward:
                    cum_t[i] += 1
            cum_c += info["collisions"]
            state, obs = next_state, next_obs
        team.end_episode()
        ep_rewards.append(tuple(cum_r)); ep_tasks.append(tuple(cum_t))
        ep_collisions.append(cum_c)

        g_run = gini_over_window(ep_rewards, ep_tasks, min(100, episode))
        with open(csv_path, "a") as f:
            f.write(f"{episode},{cum_r[0]:.2f},{cum_r[1]:.2f},"
                    f"{cum_t[0]},{cum_t[1]},{cum_c},{g_run:.4f}\n")

    # Gini per final window
    win_ginis = {w: gini_over_window(ep_rewards, ep_tasks, w) for w in WINDOWS}
    spread = max(win_ginis.values()) - min(win_ginis.values())
    verdict = "stable" if spread <= STABILITY_TOL else "oscillating"

    log.info(f"[seed {seed}] Gini per window "
             + " ".join(f"w{w}={g:.4f}" for w, g in win_ginis.items())
             + f" | spread={spread:.4f} -> {verdict.upper()}")

    return {
        "seed": seed,
        "gini_windows": {str(w): round(g, 4) for w, g in win_ginis.items()},
        "gini_official_eval": round(win_ginis[cfg["experiment_control"]["eval_last_n"]]
                                    if cfg["experiment_control"]["eval_last_n"] in win_ginis
                                    else gini_over_window(ep_rewards, ep_tasks,
                                                          cfg["experiment_control"]["eval_last_n"]), 4),
        "window_spread": round(spread, 4),
        "verdict": verdict,
        "mean_collisions": round(float(np.mean(ep_collisions[-100:])), 2),
    }


def main():
    ap = argparse.ArgumentParser(description="E2 Nash convergence validation")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--output", default="e2_convergence_validation/")
    args = ap.parse_args()

    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output)
    log = logging.getLogger(__name__)

    cfg = yaml.safe_load(open(args.config))
    episodes = cfg["experiment_control"]["episodes"] if "episodes" in cfg["experiment_control"] \
        else cfg["experiment_control"].get("max_episodes", 1000)

    log.info("=" * 60)
    log.info("E2 CONVERGENCE VALIDATION — Nash (official results left untouched)")
    log.info(f"Seeds: 0..{args.seeds-1} | Episodes: {episodes}")
    log.info(f"Stability criterion: Gini spread across windows {WINDOWS} <= {STABILITY_TOL}")
    log.info("=" * 60)

    t0 = time.time()
    results = []
    for seed in range(args.seeds):
        log.info(f"\n--- Nash | seed {seed+1}/{args.seeds} ---")
        results.append(run_seed(cfg, seed, episodes, out_dir, log))
        json.dump({"results": results}, open(out_dir / "convergence_summary.json", "w"), indent=2)

    n_stable = sum(1 for r in results if r["verdict"] == "stable")
    n_osc = sum(1 for r in results if r["verdict"] == "oscillating")
    # High-Gini (monopoly) seeds and their stability verdict
    high_gini = [r for r in results if r["gini_official_eval"] >= 0.15]
    summary = {
        "experiment": "E2_convergence_validation",
        "n_seeds": args.seeds, "n_stable": n_stable, "n_oscillating": n_osc,
        "high_gini_seeds": [{"seed": r["seed"], "gini": r["gini_official_eval"],
                             "verdict": r["verdict"]} for r in high_gini],
        "interpretation": (
            "High-Gini seeds that are STABLE indicate genuine monopoly equilibria. "
            "High-Gini seeds that are OSCILLATING would indicate non-converged policies (an artifact)."
        ),
        "results": results,
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    json.dump(summary, open(out_dir / "convergence_summary.json", "w"), indent=2)

    log.info("\n" + "=" * 60)
    log.info(f"RESULT: {n_stable} stable, {n_osc} oscillating")
    log.info("High-Gini seeds (>=0.15):")
    for r in high_gini:
        log.info(f"  seed {r['seed']}: Gini={r["gini_official_eval"]:.3f} -> {r["verdict"].upper()}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
