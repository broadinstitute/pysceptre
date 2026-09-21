#!/usr/bin/env bash
# The implementation x mechanism x core-count matrix, run inside the pinned
# container on one machine.
#
# **Ordered cheapest-first, with R's single-core CRT last.** That cell was
# still running after 53 minutes on an M4 Max when it was stopped, and an n2
# vCPU is slower, so it may take hours. Everything else finishes first, and
# capping or losing it costs no other number.
#
# Every cell runs in its own container with `--cpus` set, so the core count
# is a cgroup guarantee rather than an assumption, and in its own process, so
# peak RSS is that cell's and not a high-water mark inherited from a
# predecessor.
#
# Usage: benchmark_matrix_vm.sh <work_dir>
set -uo pipefail

WORK="${1:?usage: benchmark_matrix_vm.sh <work_dir>}"
OUT="$WORK/out"; DATA="$WORK/data"; mkdir -p "$OUT"
IMG=pysceptre-bench
MEM="--memory=48g --memory-swap=48g"

# ref -> checkout path, both shipped as worktrees of the same repo
MAIN="$WORK/ref-main/src"
BRANCH="$WORK/ref-branch/src"

pyrun() {  # pyrun <label> <src> <mechanism> <n_jobs> <cpus>
  local label="$1" src="$2" mech="$3" jobs="$4" cpus="$5"
  echo "=== $(date '+%H:%M:%S')  $label"
  sudo docker run --rm --cpus="$cpus" $MEM \
    -v "$DATA":/data -v "$OUT":/out -v "$src":/refsrc:ro \
    -e PYTHONPATH=/refsrc \
    "$IMG" python /src/scripts/benchmark_cell.py \
      "$label" /data/export_allgenes /out "$mech" "$jobs" \
    2>&1 | tail -3
}

rrun() {   # rrun <label> <step> <n_processors> <cpus>
  local label="$1" step="$2" nproc="$3" cpus="$4"
  echo "=== $(date '+%H:%M:%S')  $label"
  # Paths are the container's, not the host's: $OUT is bind-mounted at /out,
  # and benchmark_vs_r.R reads the post-QC object from its *out* dir.
  #
  # The cgroup peak is read inside the container before it exits. GNU time's
  # "Maximum resident set size" is a maximum over the process tree rather
  # than a sum, so it understates sceptre's forked workers exactly as
  # RUSAGE_SELF understated pysceptre's. Its user/sys totals do include
  # reaped children, so those stay usable for occupancy.
  sudo docker run --rm --cpus="$cpus" $MEM \
    -v "$DATA":/data -v "$OUT":/out -e N_PROCESSORS="$nproc" \
    "$IMG" bash -c "/usr/bin/time -v Rscript /src/scripts/benchmark_vs_r.R $step /data /out; \
                    echo CGROUP_PEAK_BYTES=\$(cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo 0)" \
    > "$OUT/r_${label}.log" 2>&1 || echo "  FAILED $label"
  grep -E "wall [0-9]|Maximum resident|Elapsed .wall|User time|System time|CGROUP_PEAK" \
    "$OUT/r_${label}.log" | head -6 || true
}

echo "##### phase 1: pysceptre, both refs, both mechanisms, 8 then 1 core"
pyrun branch_perm_j8 "$BRANCH" permutations 8 8
pyrun main_perm_j8   "$MAIN"   permutations 8 8
pyrun branch_crt_j8  "$BRANCH" crt          8 8
pyrun main_crt_j8    "$MAIN"   crt          8 8
pyrun branch_perm_j1 "$BRANCH" permutations 1 1
pyrun main_perm_j1   "$MAIN"   permutations 1 1
pyrun branch_crt_j1  "$BRANCH" crt          1 1
pyrun main_crt_j1    "$MAIN"   crt          1 1

echo "##### phase 2: R, everything except the long one"
rrun perm_j8 discovery_perm 8 8
rrun perm_j1 discovery_perm 1 1
rrun crt_j8  discovery      8 8

echo "##### phase 3: R single-core CRT -- the long one, deliberately last"
rrun crt_j1  discovery      1 1

echo "=== $(date '+%H:%M:%S') matrix complete"
