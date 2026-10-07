#!/usr/bin/env bash
# Check out jackkamm/fishash_analysis at the commit that produced the v2 preprint's results.
#
#   scripts/fishash_eval/fetch_upstream.sh <dest_dir>
#
# The checkout is reference material: its simulate/*.R scripts are run unmodified by
# scripts/fishash_eval/simulate.R, and nothing in it is edited. <dest_dir> must be ignored by
# git (test_data/ is), so the checkout can never be committed by accident.
set -euo pipefail

SHA=192008d7b517fe1c197daf91f7109011cd58073a
URL=https://github.com/jackkamm/fishash_analysis.git

dest=${1:?usage: fetch_upstream.sh <dest_dir>}
repo_root=$(git rev-parse --show-toplevel)

mkdir -p "$(dirname "$dest")"
if ! git -C "$repo_root" check-ignore -q "$dest"; then
    echo "refusing: $dest is not ignored by git" >&2
    exit 1
fi

if [ ! -d "$dest/.git" ]; then
    git clone --quiet --no-checkout "$URL" "$dest"
fi
git -C "$dest" fetch --quiet origin "$SHA"
git -C "$dest" -c advice.detachedHead=false checkout --quiet --detach "$SHA"

head=$(git -C "$dest" rev-parse HEAD)
if [ "$head" != "$SHA" ]; then
    echo "checkout is at $head, expected $SHA" >&2
    exit 1
fi
if [ -n "$(git -C "$dest" status --porcelain)" ]; then
    echo "checkout at $dest has local changes; it must stay unmodified" >&2
    exit 1
fi
echo "fishash_analysis at $head in $dest"
