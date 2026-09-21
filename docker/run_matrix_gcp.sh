#!/usr/bin/env bash
# The full comparison matrix on one GCP VM: R and pysceptre (two refs), both
# mechanisms, 1 and 8 cores.
#
# Why a VM rather than the laptop: the laptop is arm64 with Apple Accelerate,
# where `threadpool_limits` is inert, and it throttles on battery -- a
# comparison assembled across an evening there mixed power states. One
# machine, one session, cgroup-pinned core counts.
#
#   docker/run_matrix_gcp.sh create   # make the VM
#   docker/run_matrix_gcp.sh push     # ship the repo (both refs) and data
#   docker/run_matrix_gcp.sh run      # build the image, run the matrix
#   docker/run_matrix_gcp.sh fetch    # copy results back
#   docker/run_matrix_gcp.sh delete   # tear it down -- do not forget
set -euo pipefail

VM="${VM:-pysceptre-matrix}"
ZONE="${ZONE:-us-central1-a}"
# 16 vCPU so an 8-core cell has headroom and is not competing with the OS
# and the harness for the cores it is being measured on.
MACHINE="${MACHINE:-n2-standard-16}"
DISK="${DISK:-300GB}"
MAIN_REF="${MAIN_REF:-c98946f}"
BRANCH_REF="${BRANCH_REF:-HEAD}"
LOCAL_DATA="${LOCAL_DATA:-test_data/day0}"
# Published image whose layers a cold build reuses.
CACHE_IMG="${CACHE_IMG:-polumechanos/pysceptre-bench}"
WORK=/mnt/work
# Org projects commonly block direct SSH ingress; IAP tunnelling is the
# supported route and the only one that works here. Applies to scp too.
IAP="${IAP:---tunnel-through-iap}"

case "${1:?usage: run_matrix_gcp.sh create|push|run|fetch|delete}" in

create)
  gcloud compute instances create "$VM" --zone "$ZONE" --machine-type "$MACHINE" \
    --image-family=ubuntu-2404-lts-amd64 --image-project=ubuntu-os-cloud \
    --boot-disk-size="$DISK" --boot-disk-type=pd-ssd \
    --metadata=startup-script='#!/bin/bash
      apt-get update && apt-get install -y docker.io
      mkdir -p /mnt/work && chmod 777 /mnt/work'
  echo "created $VM ($MACHINE) in $ZONE; wait ~60s for the startup script"
  ;;

push)
  echo "shipping repo (both refs)"
  git bundle create /tmp/pysceptre.bundle --all
  gcloud compute scp $IAP /tmp/pysceptre.bundle "$VM":/tmp/ --zone "$ZONE"
  gcloud compute ssh "$VM" --zone "$ZONE" $IAP --command "set -eux
    mkdir -p $WORK && cd $WORK
    rm -rf repo ref-main ref-branch
    git clone -q /tmp/pysceptre.bundle repo
    cd repo
    git worktree add -f --detach $WORK/ref-main   $MAIN_REF
    git worktree add -f --detach $WORK/ref-branch $BRANCH_REF
    git -C $WORK/ref-main   log --oneline -1
    git -C $WORK/ref-branch log --oneline -1"
  echo "shipping data (~1.1 GB)"
  gcloud compute ssh "$VM" --zone "$ZONE" $IAP --command "mkdir -p $WORK/data $WORK/out"
  # benchmark_vs_r.R reads the post-QC object from its *out* dir, not the
  # data dir -- `postqc_fp <- file.path(out_dir, "so_postqc_crt.rds")`.
  gcloud compute scp $IAP "$LOCAL_DATA/sceptre_object.rds" "$VM":"$WORK/out/so_postqc_crt.rds" --zone "$ZONE"
  gcloud compute scp $IAP --recurse "$LOCAL_DATA/export_allgenes" "$VM":"$WORK/data/" --zone "$ZONE"
  ;;

run)
  # Build and matrix both detached: the image takes tens of minutes to build
  # (R, sceptre from source, the Python stack) and the matrix hours, so
  # neither can depend on this SSH session surviving.
  gcloud compute ssh "$VM" --zone "$ZONE" $IAP --command "
    cat > $WORK/go.sh <<'EOF'
set -eux
cd $WORK/ref-branch
# Pull the published image first so the build reuses its layers. The
# expensive ones are `rocker/r-ver` and the pinned sceptre/ondisc install
# from GitHub -- tens of minutes on a cold machine -- and they come from the
# Dockerfile, which rarely changes; what does change is the final `COPY
# scripts` layer, which rebuilds in seconds.
#
# `--cache-from` only helps if the published image carries inline cache
# metadata, which is why the push below sets BUILDKIT_INLINE_CACHE. If the
# pull fails or the metadata is absent the build just proceeds cold, so this
# is an optimisation and never a dependency.
sudo docker pull $CACHE_IMG || echo "no cached image; building cold"
DOCKER_BUILDKIT=1 sudo docker build -f docker/Dockerfile \
  --cache-from $CACHE_IMG --build-arg BUILDKIT_INLINE_CACHE=1 \
  -t pysceptre-bench .
sudo docker run --rm pysceptre-bench cat /src/ENVIRONMENT.txt > $WORK/out_env.txt || true
bash scripts/benchmark_matrix_vm.sh $WORK
EOF
    chmod +x $WORK/go.sh
    nohup bash $WORK/go.sh > $WORK/matrix.log 2>&1 &
    echo launched"
  echo "running in the background; follow with: $0 tail"
  ;;

tail)
  gcloud compute ssh "$VM" --zone "$ZONE" $IAP --command "tail -40 $WORK/matrix.log"
  ;;

fetch)
  DEST="${DEST:-test_data/day0/matrix_gcp}"
  mkdir -p "$DEST"
  gcloud compute scp $IAP --recurse "$VM":"$WORK/out/*" "$DEST"/ --zone "$ZONE" || true
  gcloud compute scp $IAP "$VM":"$WORK/matrix.log" "$DEST"/ --zone "$ZONE" || true
  gcloud compute scp $IAP "$VM":"$WORK/out_env.txt" "$DEST"/ --zone "$ZONE" || true
  echo "fetched -> $DEST"
  ;;

push-image)
  # Requires `docker login` on the VM: credentials in a local keychain
  # (credsStore) cannot be shipped. Tags with the branch sha so a stale
  # cache is identifiable.
  gcloud compute ssh "$VM" --zone "$ZONE" $IAP --command "set -eux
    cd $WORK/ref-branch
    sha=\$(git rev-parse --short HEAD)
    sudo docker tag pysceptre-bench $CACHE_IMG:latest
    sudo docker tag pysceptre-bench $CACHE_IMG:\$sha
    sudo docker push $CACHE_IMG:latest
    sudo docker push $CACHE_IMG:\$sha"
  ;;

delete)
  gcloud compute instances delete "$VM" --zone "$ZONE" --quiet
  echo "deleted $VM. Confirm no disk is orphaned:"
  gcloud compute disks list --filter="name~$VM" --zones="$ZONE" || true
  ;;
esac
