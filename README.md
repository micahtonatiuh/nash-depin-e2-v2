# Nash-DePIN — Experiment E2: Nash vs QMIX/VDN (Fairness)

> **Paper:** "Fair Task Allocation in Physical Multi-Robot Systems via Nash Q-Learning: A Comparison with Value Decomposition Methods" (submitted) — central contribution
> **Goal:** Show that Nash Q-Learning distributes tasks more equitably than QMIX/VDN in mixed-motive MRTA.

## Quick start

```bash
git clone <repo_url>
cd nash-depin-e2
pip install -r requirements.txt          # includes torch for QMIX/VDN
pytest tests/test_baselines.py -v        # 20 tests should pass

# Run the three algorithms (10 seeds each)
python3 experiments/run_e2.py --algo nash --seeds 10 --output e2_results/nash/
python3 experiments/run_e2.py --algo qmix --seeds 10 --output e2_results/qmix/
python3 experiments/run_e2.py --algo vdn  --seeds 10 --output e2_results/vdn/

# Primary statistical analysis (reported in the paper)
python3 experiments/analyze_e2_stats_v2.py --results_dir e2_results/

# Figures and descriptive table
python3 scripts/analyze_e2.py --results e2_results/ --output e2_results/plots/
```

Runs are deterministic given the seed: `--seeds 10` uses seeds 0–9.

## Structure

```
nash-depin-e2/
├── config/e2_config.yaml               ← Mixed rewards, seeds, hypothesis thresholds
├── nash_depin/
│   ├── environment_mixed.py            ← Gridworld with MIXED rewards (+10/-5/+2)
│   ├── fairness_metrics.py             ← Gini, Jain's index, task balance
│   ├── nash_solver.py                  ← Bimatrix Nash solver + egalitarian refinement
│   ├── nash_agent_e2.py                ← Nash team with PER-AGENT rewards
│   └── equilibrium_refinement.md       ← Method note on equilibrium selection
├── baselines/
│   ├── qmix_baseline.py                ← QMIX (monotonic mixing network, PyTorch)
│   └── vdn_baseline.py                 ← VDN (additive decomposition, PyTorch)
├── experiments/
│   ├── run_e2.py                       ← Orchestrator: 1 algorithm × N seeds
│   ├── analyze_e2_stats_v2.py          ← PRIMARY analysis (median, Mann-Whitney, Cliff's delta)
│   ├── validate_e2_convergence.py      ← Stability check on the high-Gini seeds
│   └── export_e2_qtables_for_e5.py     ← Exports Nash Q-tables for the hardware runs
├── scripts/analyze_e2.py               ← Figures + descriptive comparison table
├── tests/test_baselines.py             ← 20 unit tests
├── e2_stats_report.json                ← Output of the primary analysis (reported)
└── e2_convergence_validation/          ← Per-seed trajectories + stability verdicts
```

## Reward structure

The environment returns **per-agent** rewards combining competitive and
cooperative components:

| Component | Value | Type |
|---|---|---|
| Task completion | +10 (one robot only) | competitive |
| Collision | −5 (both robots) | cooperative |
| Spatial cooperation | +2 (both robots) | cooperative |

Nash uses each agent's individual reward; QMIX and VDN use the sum (team
reward), which is what their formulations require.

## Results as reported

From `e2_stats_report.json` (10 seeds per algorithm, metrics averaged over the
final 100 episodes of each run):

| Algorithm | Gini median | Gini IQR | Seeds with Gini < 0.15 |
|---|---|---|---|
| Nash Q-Learning | 0.079 | [0.040, 0.285] | 7/10 |
| QMIX | 0.372 | [0.371, 0.374] | 0/10 |
| VDN | 0.369 | [0.366, 0.373] | 0/10 |

Mann-Whitney U (one-sided, Holm-corrected): Nash vs QMIX p = 5.0e-4, Cliff's
δ = −0.88; Nash vs VDN p = 4.4e-4, Cliff's δ = −0.94. Both large effects.

**The median is the primary statistic, not the mean.** The Nash Gini
distribution is bimodal: 3 of 10 seeds converge to monopoly equilibria (Gini
≈ 0.34–0.35). These were verified to be genuine stable equilibria rather than
non-converged policies — `validate_e2_convergence.py` reports 10/10 seeds
stable across evaluation windows of 50, 100 and 200 episodes. The bimodality is
reported as a finding; see `nash_depin/equilibrium_refinement.md`.

The mean Nash Gini (0.147) sits close to the preregistered 0.15 threshold and
is sensitive to individual seeds, which is why the non-parametric analysis in
`experiments/analyze_e2_stats_v2.py` governs. The parametric statistics printed
by `scripts/analyze_e2.py` are descriptive only.

## Runtime

- **Nash:** ~6 min per seed (fast solver). 10 seeds ≈ 60 min.
- **QMIX/VDN:** ~3–5 min per seed (small networks, CPU). 10 seeds ≈ 30–50 min.
- All three algorithms: ~2–3 hours.
