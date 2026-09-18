#!/usr/bin/env bash
# Drive the per-step benchmark. Each step runs as its own process under
# /usr/bin/time -l so peak RSS is measured per step, not as a whole-pipeline
# high-water mark.
#
# Usage: benchmark_vs_r.sh <data_dir> <out_dir> [step ...]
set -uo pipefail

DATA_DIR="${1:?usage: benchmark_vs_r.sh <data_dir> <out_dir> [step ...]}"
OUT_DIR="${2:?usage: benchmark_vs_r.sh <data_dir> <out_dir> [step ...]}"
shift 2
STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(prepare discovery calibration power)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT_DIR"

for step in "${STEPS[@]}"; do
  log="$OUT_DIR/r_${step}.log"
  echo "=== R: $step -> $log"
  /usr/bin/time -l Rscript "$HERE/benchmark_vs_r.R" "$step" "$DATA_DIR" "$OUT_DIR" > "$log" 2>&1
  status=$?
  # /usr/bin/time -l writes to stderr, which lands in the log alongside R's output.
  peak=$(grep -E "maximum resident set size" "$log" | awk '{printf "%.2f", $1/1e9}')
  real=$(grep -E "^\s+[0-9.]+ real" "$log" | awk '{print $1}')
  if [ $status -ne 0 ]; then
    echo "    FAILED (exit $status) -- see $log"
    grep -iE "^Error|Execution halted" "$log" | head -3
  else
    echo "    ok: ${real}s process wall, ${peak} GB peak RSS"
  fi
done
echo "=== done -> $OUT_DIR"
