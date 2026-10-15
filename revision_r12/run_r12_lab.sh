#!/usr/bin/env bash
# run_r12_lab.sh — R1.2 / R2.3 Kalai ablation on the lab PC.
# Run from the nash-depin-e2-v2 repository root (needs qtables_all_seeds/ and reeval/ from v2.1).
#   bash revision_r12/run_r12_lab.sh            # full: 3 variants x 20 seeds (config seeds_contingency)
#   SEEDS="0 1 2" CHUNKS=3 bash revision_r12/run_r12_lab.sh   # custom seeds / processes per variant
#   bash revision_r12/run_r12_lab.sh --quick    # control only: egalitarian seed 0 (~6-12 min)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
OUT="revision_r12/results_lab_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT/logs"
touch revision_r12/__init__.py reeval/__init__.py
{
  echo "host: $(hostname)  date: $(date -Is)"
  $PY -c "import sys,numpy;print('python',sys.version.split()[0],'numpy',numpy.__version__)"
  grep -m1 "model name" /proc/cpuinfo || true
  sha256sum revision_r12/kalai_ablation.py nash_depin/nash_agent_e2.py nash_depin/nash_solver.py \
            nash_depin/environment_mixed.py reeval/reeval_frozen_e2.py config/e2_config.yaml
} | tee "$OUT/env.txt"

if [[ "${1:-}" == "--quick" ]]; then
  $PY revision_r12/kalai_ablation.py --variant egalitarian --seeds 0 --out "$OUT" | tee "$OUT/logs/egalitarian.log"
else
  SEEDS=${SEEDS:-$(seq -s ' ' 0 19)}; CHUNKS=${CHUNKS:-2}
  read -ra S <<< "$SEEDS"
  for v in egalitarian utilitarian single; do
    for ((c=0; c<CHUNKS; c++)); do
      part=(); for ((k=c; k<${#S[@]}; k+=CHUNKS)); do part+=("${S[k]}"); done
      [[ ${#part[@]} -gt 0 ]] && $PY revision_r12/kalai_ablation.py --variant $v --seeds "${part[@]}" \
          --out "$OUT" > "$OUT/logs/${v}_$c.log" 2>&1 &
    done
  done
  echo "$((3*CHUNKS)) processes running; progress: tail -f $OUT/logs/*.log"
  wait
fi
$PY revision_r12/summarize.py --results "$OUT" | tee "$OUT/summary.txt"
(cd "$OUT" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS)
echo "DONE -> $OUT"
