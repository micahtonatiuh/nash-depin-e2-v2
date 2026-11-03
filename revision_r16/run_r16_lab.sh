#!/usr/bin/env bash
# R1.6 scalability — lab PC runner. Run from the root of nash-depin-e2-v2.
#   bash revision_r16/run_r16_lab.sh --check   # controls only (~8 min): env equivalence + Nash E2 bit-exact
#   bash revision_r16/run_r16_lab.sh           # 3 scenarios x 4 algorithms x 10 seeds, JOBS parallel processes
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}; JOBS=${JOBS:-12}; SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OUT="revision_r16/results_lab_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$OUT/logs"
{ hostname; date -Is
  $PY -c "import sys,numpy,torch;print('python',sys.version.split()[0],'numpy',numpy.__version__,'torch',torch.__version__)"
  grep -m1 "model name" /proc/cpuinfo || true
  sha256sum revision_r16/r16_scalability.py nash_depin/environment_mixed.py nash_depin/nash_solver.py \
            nash_depin/nash_agent_e2.py baselines/qmix_baseline.py baselines/vdn_baseline.py config/e2_config.yaml; } | tee "$OUT/env.txt"
$PY revision_r16/r16_scalability.py --check | tee "$OUT/check.txt"
[[ "${1:-}" == "--check" ]] && exit 0
# one job per (scenario, algo, seed); longest (7x7) first
for scn in S1 S3 S2; do for algo in nash nash_util qmix vdn; do for s in $SEEDS; do echo "$scn $algo $s"; done; done; done > "$OUT/jobs.txt"
echo "$(wc -l < "$OUT/jobs.txt") jobs, $JOBS in parallel; progress: tail -f $OUT/logs/*.log"
xargs -P "$JOBS" -L 1 bash -c '$0 revision_r16/r16_scalability.py --scenario $1 --algo $2 --seeds $3 --out '"$OUT"' >> '"$OUT"'/logs/$1_$2.log 2>&1' "$PY" < "$OUT/jobs.txt"
cat "$OUT"/logs/*.log
$PY revision_r16/r16_summary.py "$OUT" | tee "$OUT/summary.txt"
(cd "$OUT" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS)
echo "DONE -> $OUT"
