# Release v2.1 — evidence for the Paper 1 correction (IEEE Access)

Adds the material cited in the corrected manuscript (Sec. III-B, III-E, IV-B, IV-C, V-E and Data Availability).

## Contents
- `qtables_all_seeds/` — final E2 Nash Q-tables, seeds 0–9 (`seed_{s}_agent_{i}.npz`), with SHA-256 checksums in `manifest.json`.
  `fair_seeds.json` corrected: fair = 0,1,2,3,4,7,9; monopolistic = 5,6,8 (the previous file listed all ten seeds under a "fair" label).
- `reeval/reeval_frozen_e2.py` — frozen-policy re-evaluation (Sec. IV-B "Mechanism check", IV-C, V-E).
- `reeval/results_reeval.json`, `reeval/reeval_stdout.txt` — outputs of the script.
- `reeval/audit_equilibria_fidelity.py`, `reeval/results_audit.json` — exploitability of the solver's selected profiles (Sec. III-C) and match probabilities under mixed-strategy sampling (Sec. V-D).
- `e2_results/task_level_fairness.json` — task shares, task-based Gini, collisions per seed and method, computed from the existing `summary.json` files (Table 1 "Top share" column).

## Reproduce
From the repository root (numpy 2.x):

    python3 reeval/reeval_frozen_e2.py

Expected summary (fair = pooled Gini < 0.15):

    e2             sample_eps median=0.0822 fair=[0, 1, 2, 3, 4, 7, 9]
    e2             argmax_eps median=0.0839 fair=[0, 1, 2, 3, 4, 7, 9]
    e2_transposed  sample_eps median=0.1032 fair=[0, 1, 2, 3, 4, 7, 9]
    e2_transposed  argmax_eps median=0.1182 fair=[0, 1, 2, 3, 4, 7, 9]
    physical       sample_eps median=0.1210 fair=[0, 2, 4, 7, 9]
    physical       argmax_eps median=0.1123 fair=[0, 2, 3, 4, 7, 9]

Audit (Sec. III-C, V-D):

    python3 reeval/audit_equilibria_fidelity.py
    # relative exploitability: median 0.0014, p99 0.0025; match 12/200 steps: seed 2 0.941/0.871, seed 4 0.975/0.909

## Notes
- E2 training and evaluation start at R0=(0,0), R1=(4,4) (`environment_mixed.py`, `reset()`); the physical experiments (E3, E4) start at R0=(4,0), R1=(0,4).
- During training, Nash agents sample from their equilibrium strategy sigma_i with probability 1-epsilon (`nash_agent_e2.py`, `select_action`); deployment uses argmax sigma_i at epsilon=0.
