# revision_r12 — Kalai (egalitarian) selection ablation, R1.2 / R2.3

Drop this folder into the root of a nash-depin-e2-v2 (v2.1) clone. Only new files; nothing in the repo is modified
(the solver used by `NashTeamE2.learn` is swapped at run time, see `kalai_ablation.py`).

Variants (everything else identical to E2: env, hyperparameters, 1000 episodes, seeds):
- `egalitarian` — joint_nash(fast=True, equity=True): 4 best-response starts, min |V0−V1| (paper E2; regression control)
- `utilitarian` — same 4 starts, max V0+V1 (changes only the selection rule)
- `single` — joint_nash(fast=True, equity=False): 1 start, i.e. what USE_EQUITY=False does in the code

Lab PC:
    bash revision_r12/run_r12_lab.sh --quick   # gate: egalitarian seed 0 must print "identical to qtables_all_seeds = True"
    bash revision_r12/run_r12_lab.sh           # 3 variants x seeds 0-19, 2 processes per variant (CHUNKS=2)

Note: E2 training is bit-reproducible only on the same CPU/BLAS kernel path (OPENBLAS_CORETYPE changes the
Q-tables after ~120 episodes). The cloud preview (results_cloud_preview/) is therefore statistically, not
bitwise, comparable; the lab run is the reference.
