# R1.1 — Lab-PC replication package

Paper: *Fair Task Allocation in Physical Multi-Robot Systems via Nash Q-learning* (IEEE Access, Access-2026-41887), revision for Reviewer 1, Comment 1.

## What it does

It reruns on the lab PC every number behind the R1.1 response, and compares each one with the cloud reference run (`expected/`):

| Step | Script | What it supports in the paper |
|---|---|---|
| 2 | `exact_team_dp.py` | Proposition 1 (team value 258.81 for monopoly and alternation), Proposition 2 (matrix 280/266/266/280), Nash remark |
| 2 | `projection_check.py` | Corollary (additive projection selects *hold* at ε = 1) |
| 2 | `extra_checks.py` | Closure of the 14-reward set; physical start (240.07) |
| 3 | `team_q_no_igm.py` (1000 ep, seeds 0–19) | Team learner at the E2 budget |
| 4 | `team_q_no_igm.py` (10 000 ep, seeds 0–19) | Empirical check of Proposition 1 |

The original lab code is never modified: every script only imports `nash_depin` from the repo.

## How to run

```bash
cd ~
git clone https://github.com/micahtonatiuh/nash-depin-e2-v2.git   # if not present
git -C ~/nash-depin-e2-v2 checkout v2.1
# copy this folder to the lab PC, then:
bash r1_1_lab/run_r1_1_lab.sh ~/nash-depin-e2-v2 ~/nash-depin-e2
```

Requirements: Python 3, numpy, pyyaml (already used by E2). Duration: about 20–25 minutes.

## What to check

1. `provenance.txt`: commit `fb8f4c1…` (tag v2.1) and the three SHA-256 hashes match the expected ones.
2. Step 1 (`ast_check.py`): the environment and fairness code of the original lab folder `~/nash-depin-e2` is logically identical to the public repo (comments and docstrings ignored).
3. `comparison.txt`: **ALL PASS** means the lab reproduces the cloud results number by number.

If anything differs, do not overwrite either set: keep both folders and report the differing values. The team learner uses NumPy's legacy seeded RNG, so identical results are expected across machines; a NumPy version change could alter floating-point details, which `provenance.txt` records.

## Output

`results_lab/<timestamp>/` with all JSON and logs, `comparison.txt`, and `SHA256SUMS` for archiving with the revision.

## v1.1 changes (9 Oct 2026, after the first lab run)

- **Tied labels.** Against a wall, `up`, `left` and `stay` are the same move, so their fitted values are exact ties and floating-point noise picks the label. The scripts now also report the effective outcome (`hold` / `handover`), and the comparison checks the outcome, not the label.
- **Original vs public environment.** `ast_check.py` now has two levels (L1 logic, L2 logic with all text masked) and writes a diff file when they differ. `env_equivalence.py` compares the behaviour of both environments step by step over 400,000 steps covering all 15,000 state–action pairs. Behaviour is the decisive test.
- **`--quick` mode** reruns steps 0–2 and 5 in about 2 minutes, without the team learner (already reproduced 40/40 in the first run).

```bash
bash r1_1_lab/run_r1_1_lab.sh ~/nash-depin-e2-v2 ~/nash-depin-e2 --quick
```
