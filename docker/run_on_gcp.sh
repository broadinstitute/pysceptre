#!/usr/bin/env bash
# Run the benchmark inside the pinned container on a GCP VM.
#
# Why GCP rather than the laptop: the laptop is arm64 and its numpy links
# Apple Accelerate, which threadpoolctl cannot introspect -- so
# threadpool_limits is a no-op there and the code path the paper describes is
# never exercised. x86_64 Linux with OpenBLAS is what users and CI see.
#
# Data is already in GCS, so the VM pulls inputs directly and pushes results
# back; nothing large crosses the local machine.
#
# Usage:
#   docker/run_on_gcp.sh create     # make the VM
#   docker/run_on_gcp.sh push       # ship the working tree (repo is private)
#   docker/run_on_gcp.sh data       # ship the inputs
#   docker/run_on_gcp.sh run [steps...]
#   docker/run_on_gcp.sh fetch      # copy results out of GCS
#   docker/run_on_gcp.sh delete     # tear the VM down (do not forget)
set -euo pipefail

VM="${VM:-pysceptre-bench}"
ZONE="${ZONE:-us-central1-a}"              # same region as the data
MACHINE="${MACHINE:-n2-standard-8}"        # 8 vCPU / 32 GB; consistent clocks,
                                           # and headroom for --memory sweeps
DISK="${DISK:-200GB}"
# No defaults: these name a specific screen, and a wrong-but-plausible
# default is worse than a missing one when the output is a benchmark.
GCS_IN="${GCS_IN:?set GCS_IN to the GCS prefix holding the ondisc inputs}"
GCS_SO="${GCS_SO:?set GCS_SO to the GCS path of the sceptre_object.rds}"
REPO="${REPO:-https://github.com/broadinstitute/pysceptre.git}"
REF="${REF:-main}"

case "${1:?usage: run_on_gcp.sh create|run|fetch|delete}" in

create)
  gcloud compute instances create "$VM" --zone "$ZONE" --machine-type "$MACHINE" \
    --image-family ubuntu-2404-lts-amd64 --image-project ubuntu-os-cloud \
    --boot-disk-size "$DISK" --boot-disk-type pd-balanced \
    --scopes storage-rw \
    --metadata-from-file startup-script=<(cat <<'STARTUP'
#!/bin/bash
set -eux
apt-get update && apt-get install -y docker.io git
systemctl enable --now docker
STARTUP
)
  echo "created $VM in $ZONE ($MACHINE). Wait ~60s for the startup script, then: $0 run"
  ;;

push)
  # The repository is private, so the VM cannot clone it. Ship the working
  # tree instead, which also means the benchmark measures exactly the code
  # under test rather than whatever a remote ref happens to hold.
  TARBALL=$(mktemp -t pysceptre-src).tar.gz
  git -C "$(git rev-parse --show-toplevel)" archive --format=tar.gz -o "$TARBALL" HEAD
  echo "shipping $(git rev-parse --short HEAD) ($(du -h "$TARBALL" | cut -f1))"
  gcloud compute scp "$TARBALL" "$VM":/tmp/pysceptre-src.tar.gz --zone "$ZONE"
  gcloud compute ssh "$VM" --zone "$ZONE" --command "set -eux
    sudo mkdir -p /mnt/work && sudo chown \$(id -u):\$(id -g) /mnt/work
    rm -rf /mnt/work/pysceptre && mkdir -p /mnt/work/pysceptre
    tar -xzf /tmp/pysceptre-src.tar.gz -C /mnt/work/pysceptre
    echo '$(git rev-parse HEAD)' > /mnt/work/pysceptre/COMMIT
  "
  rm -f "$TARBALL"
  ;;

data)
  # The VM's default service account cannot read the inputs bucket, and
  # granting it access would mean changing project IAM for a benchmark. Ship
  # the local copy instead.
  LOCAL_DATA="${LOCAL_DATA:?set LOCAL_DATA to the directory holding gene.odm, grna.odm, sceptre_object.rds, positive.rds}"
  gcloud compute ssh "$VM" --zone "$ZONE" --command "mkdir -p /mnt/work/data /mnt/work/out"
  for f in gene.odm grna.odm sceptre_object.rds positive.rds; do
    echo "  -> $f"
    gcloud compute scp "$LOCAL_DATA/$f" "$VM":/mnt/work/data/"$f" --zone "$ZONE"
  done
  ;;

run)
  shift || true
  STEPS="${*:-prepare discovery calibration power}"
  gcloud compute ssh "$VM" --zone "$ZONE" --command "set -eux
    cd /mnt/work/pysceptre
    mkdir -p /mnt/work/data /mnt/work/out
    cp COMMIT /mnt/work/out/COMMIT
    sudo docker build -f docker/Dockerfile -t pysceptre-bench .
    sudo docker run --rm pysceptre-bench cat /src/ENVIRONMENT.txt > /mnt/work/out/ENVIRONMENT.txt
    # --cpus=1 makes single-core a cgroup guarantee, not an assumption.
    for step in $STEPS; do
      echo \"=== \$step\"
      sudo docker run --rm --cpus=1 --memory=24g --memory-swap=24g \
        -v /mnt/work/data:/data -v /mnt/work/out:/out pysceptre-bench \
        /usr/bin/time -v Rscript /src/scripts/benchmark_vs_r.R \"\$step\" /data /out \
        > /mnt/work/out/r_\$step.log 2>&1 || echo \"  FAILED \$step\"
      grep -E 'Maximum resident|Elapsed .wall' /mnt/work/out/r_\$step.log | head -2 || true
    done
  "
  ;;

fetch)
  DEST="${DEST:-test_data/bench_gcp}"
  mkdir -p "$DEST"
  gcloud compute scp --recurse "$VM":/mnt/work/out/'*' "$DEST"/ --zone "$ZONE"
  echo "fetched -> $DEST"
  ;;

delete)
  gcloud compute instances delete "$VM" --zone "$ZONE" --quiet
  ;;

*) echo "unknown: $1"; exit 1 ;;
esac
