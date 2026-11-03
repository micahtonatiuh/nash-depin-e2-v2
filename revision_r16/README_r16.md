# revision_r16 — R1.6 scalability (Paper 1, Access-2026-41887)

New files only; `nash_depin/`, `baselines/`, `experiments/` and `config/` are imported, never modified.

| Scenario | Grid | Tasks | Purpose | States (Nash) | Episodes |
|---|---|---|---|---|---|
| E2ref | 5x5 | 1 station (0,0) +10 | control, must equal E2 | 625 | 1000 |
| S1 | 7x7 | 1 station (0,0) +10 | larger arena, longer travel | 2401 | 3000 |
| S2 | 5x5 | 2 simultaneous stations, +10 and +5 | as many tasks as robots | 625 | 1000 |
| S3 | 5x5 + shelf (2,1)-(2,3) | 1 job at a time, random among (0,0), (0,4), (4,0) | dynamic job arrivals, factory layout | 1875 | 2000 |

E2 rewards (−5 collision, +2 cooperation), hyperparameters and training loop of `experiments/run_e2.py`.
Epsilon decay rescaled so that epsilon reaches 0.05 at the same fraction of training.
Evaluation as in R1.3: frozen policy, 100 episodes, eps = 0.05 (Nash samples sigma, QMIX/VDN argmax).

## Lab PC
```bash
cd ~/nash-depin-e2-v2
mkdir -p revision_r16 && cp <copied files> revision_r16/
bash revision_r16/run_r16_lab.sh --check   # ~8 min: env control PASS + Nash E2ref seed 0 bit-exact vs qtables_all_seeds
bash revision_r16/run_r16_lab.sh           # 120 jobs (3 scenarios x 4 algorithms x 10 seeds), JOBS=12 in parallel
```
Expected: both controls PASS. Full run ~4 h with 12 parallel jobs (7x7 jobs ~25 min each).
Output: `revision_r16/results_lab_<timestamp>/` with per-seed JSON, Q-tables / agent weights, logs,
`summary.txt` and `SHA256SUMS`. Send `summary.txt` and `check.txt`.
