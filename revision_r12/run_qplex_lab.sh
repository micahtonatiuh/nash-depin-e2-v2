#!/usr/bin/env bash
# run_qplex_lab.sh — R1.2 QPLEX baseline on the lab PC (run from the nash-depin-e2-v2 root; needs torch).
#   bash revision_r12/run_qplex_lab.sh --quick   # control: QMIX seed 0 with this harness vs e2_results/qmix (+ QPLEX seed 0)
#   bash revision_r12/run_qplex_lab.sh           # QPLEX seeds 0-19, CHUNKS processes (default 10)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
OUT="revision_r12/results_qplex_lab_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$OUT/logs"
touch revision_r12/__init__.py reeval/__init__.py
{ hostname; date -Is
  $PY -c "import sys,numpy,torch;print('python',sys.version.split()[0],'numpy',numpy.__version__,'torch',torch.__version__)"
  grep -m1 "model name" /proc/cpuinfo || true
  sha256sum revision_r12/qplex_run.py revision_r12/qplex_baseline.py baselines/qmix_baseline.py \
            nash_depin/environment_mixed.py config/e2_config.yaml; } | tee "$OUT/env.txt"
if [[ "${1:-}" == "--quick" ]]; then
  $PY revision_r12/qplex_run.py --algo qmix  --seeds 0 --out "$OUT" > "$OUT/logs/qmix_0.log" 2>&1 &
  $PY revision_r12/qplex_run.py --algo qplex --seeds 0 --out "$OUT" > "$OUT/logs/qplex_0.log" 2>&1 &
  wait; cat "$OUT"/logs/*.log
else
  SEEDS=${SEEDS:-$(seq -s ' ' 0 19)}; CHUNKS=${CHUNKS:-10}; read -ra S <<< "$SEEDS"
  for ((c=0; c<CHUNKS; c++)); do
    part=(); for ((k=c; k<${#S[@]}; k+=CHUNKS)); do part+=("${S[k]}"); done
    [[ ${#part[@]} -gt 0 ]] && $PY revision_r12/qplex_run.py --algo qplex --seeds "${part[@]}" --out "$OUT" \
        > "$OUT/logs/qplex_$c.log" 2>&1 &
  done
  echo "$CHUNKS processes running; progress: tail -f $OUT/logs/*.log"; wait; cat "$OUT"/logs/*.log
fi
(cd "$OUT" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS)
echo "DONE -> $OUT"
