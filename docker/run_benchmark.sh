#!/usr/bin/env bash
# Run the single-core comparison inside the pinned Linux image.
#
# --cpus=1 makes "single core" a cgroup guarantee rather than an assumption
# about what the process happens to do, and --memory makes the constrained-RAM
# question ("how long with 8 GB?") a real experiment instead of an estimate:
# the kernel enforces it, and an over-budget run is OOM-killed rather than
# quietly swapping.
set -euo pipefail

DATA_DIR="${DATA_DIR:?set DATA_DIR to the host directory holding the moi5 inputs}"
OUT_DIR="${OUT_DIR:-$DATA_DIR/bench_linux}"
CPUS="${CPUS:-1}"
MEMORY="${MEMORY:-16g}"
IMAGE="${IMAGE:-pysceptre-bench}"
STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(prepare discovery calibration power)

mkdir -p "$OUT_DIR"
echo "image=$IMAGE cpus=$CPUS memory=$MEMORY steps=${STEPS[*]}"

docker run --rm \
  --cpus="$CPUS" --memory="$MEMORY" --memory-swap="$MEMORY" \
  -v "$(cd "$DATA_DIR" && pwd)":/data \
  -v "$(cd "$OUT_DIR" && pwd)":/out \
  "$IMAGE" bash -lc '
    cat /src/ENVIRONMENT.txt 2>/dev/null || true
    for step in '"${STEPS[*]}"'; do
      echo "=== R: $step"
      /usr/bin/time -v Rscript /src/scripts/benchmark_vs_r.R "$step" /data /out \
        > "/out/r_${step}.log" 2>&1 || echo "  FAILED (see r_${step}.log)"
      grep -E "Maximum resident|Elapsed" "/out/r_${step}.log" | head -2 || true
    done
    if [ -d /out/export ]; then
      echo "=== pysceptre: discovery"
      /usr/bin/time -v python /src/scripts/benchmark_pysceptre.py discovery /out/export /out \
        > /out/pysceptre_discovery.log 2>&1 || echo "  FAILED"
      grep -E "Maximum resident|Elapsed" /out/pysceptre_discovery.log | head -2 || true
    fi
  '
echo "done -> $OUT_DIR"
