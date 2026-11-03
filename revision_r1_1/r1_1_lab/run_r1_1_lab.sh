#!/usr/bin/env bash
# =====================================================================
# R1.1 — lab-PC replication run (Paper 1 revision, Access-2026-41887)
# Runs the exact analysis and the unfactorized team learner against the
# public repo at tag v2.1, logs the environment, and compares every number
# with the cloud reference outputs in expected/.
# Original lab code is NOT modified: the repo is only imported.
#
# Usage:  bash run_r1_1_lab.sh [PUBLIC_REPO] [ORIGINAL_LAB_REPO] [--quick]
#   defaults: ~/nash-depin-e2-v2   ~/nash-depin-e2
#   --quick : steps 0-2 and 5 only (about 2 min); skips the team learner
# Duration: ~20 min full (the two 10k-episode shards run in parallel).
# =====================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${1:-$HOME/nash-depin-e2-v2}"
ORIG="${2:-$HOME/nash-depin-e2}"
QUICK="${3:-}"
OUT="$HERE/results_lab/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
cd "$HERE"

echo "== 0. Environment and provenance ==" | tee "$OUT/provenance.txt"
{
  echo "date: $(date -Is)"; echo "host: $(hostname)"; uname -a
  lsb_release -ds 2>/dev/null || true
  python3 -c "import sys,numpy,yaml;print('python',sys.version.split()[0],'| numpy',numpy.__version__,'| pyyaml',yaml.__version__)"
  echo "repo: $REPO"; git -C "$REPO" rev-parse HEAD; git -C "$REPO" describe --tags --always
  echo "expected commit (cloud run): fb8f4c1b10af3d14d2273bf15df32a0f4f8d6919 (tag v2.1)"
  sha256sum "$REPO/nash_depin/environment_mixed.py" "$REPO/nash_depin/fairness_metrics.py" "$REPO/config/e2_config.yaml"
  echo "expected sha256 (cloud run):"
  echo "  5ae65aa72239d55cdd53124717a5f874c724a569a357a179ee8cb8bf3f16d0c6  environment_mixed.py"
  echo "  02fd38925eabf4a08dee5d941c058716602b9387ca3ba0123b6534fcb2b5c3a9  fairness_metrics.py"
  echo "  c94e765fe3bdabe3697d13c8b5e23566c50bf4e9cc85be12762073f91deecd56  e2_config.yaml"
  sha256sum "$HERE"/*.py
} | tee -a "$OUT/provenance.txt"

echo "== 1. Original lab code vs public repo (logic only) ==" | tee -a "$OUT/provenance.txt"
python3 ast_check.py "$REPO" "$ORIG" "$OUT" | tee -a "$OUT/provenance.txt" || true
echo "== 1b. Original lab environment vs public repo (behaviour, all state-action pairs) ==" | tee -a "$OUT/provenance.txt"
python3 env_equivalence.py "$REPO" "$ORIG" 2000 | tee -a "$OUT/provenance.txt" || true

echo "== 2. Exact analysis (seconds) =="
python3 exact_team_dp.py --repo "$REPO" --out "$OUT/exact_team_dp_results.json" | tee "$OUT/exact_team_dp.log"
python3 projection_check.py "$REPO" | tee "$OUT/projection_check.log"
python3 extra_checks.py "$REPO"     | tee "$OUT/extra_checks.log"

if [ "$QUICK" != "--quick" ]; then
echo "== 3. Team learner, E2 budget (1000 episodes, seeds 0-19, ~2 min) =="
python3 team_q_no_igm.py --repo "$REPO" --seeds 0-19 --out "$OUT/team_q_1k_s0-19.json" | tee "$OUT/team_q_1k.log"

echo "== 4. Team learner, 10000 episodes (seeds 0-9 and 10-19 in parallel, ~10 min) =="
python3 team_q_no_igm.py --repo "$REPO" --seeds 0-9   --episodes 10000 --out "$OUT/team_q_10k_s0-9.json"   > "$OUT/team_q_10k_s0-9.log" &
python3 team_q_no_igm.py --repo "$REPO" --seeds 10-19 --episodes 10000 --out "$OUT/team_q_10k_s10-19.json" > "$OUT/team_q_10k_s10-19.log" &
wait
grep "^seed" "$OUT"/team_q_10k_s*.log
fi

echo "== 5. Comparison with the cloud reference =="
python3 compare_results.py "$OUT" "$HERE/expected" | tee "$OUT/comparison.txt"
( cd "$OUT" && sha256sum * > SHA256SUMS )
echo "Results in: $OUT"
