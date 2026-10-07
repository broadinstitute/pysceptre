#!/usr/bin/env python
"""Run crispat's Gaussian mixture guide assignment (`ga_gauss`) on the simulated datasets.

Usage:
    python scripts/fishash_eval/run_crispat.py <sims_dir> <scenario> <runs_dir>
        [--block START STEP] [sim_label ...]

Run it with the evaluation venv's Python (test_data/fishash_eval/venvs/eval), which pins
crispat to jackkamm/crispat@bd2efa7, the fork the fishash paper ran, and numpy, pandas and
scipy to the minor versions the paper's setup_env.sh pins (2.3, 2.3, 1.16); that crispat
raises a TypeError under numpy 2.5. With no sim_label, every dataset in the scenario's
manifest is run, in manifest order.

crispat gets the input the paper's bin/convert_to_anndata.R built: a cells x guides CSC matrix
of int64 counts (R's t() of a dgCMatrix arrives through reticulate as CSC, then X.astype(int)),
obs_names cell_1..N, var_names feature_1..G and an int32 obs["batch"] of 0. It is called as
the paper's bin/run_crispat_gauss.py calls it, nonzero=True and defaults otherwise. --block
START STEP adds start_gRNA=START and step=STEP, so one call covers 0-based guides START to
START + STEP - 1. crispat.gauss.plot_loss is replaced by a no-op (one PNG per guide, and the
paper deleted them); the fork already disables plot_fitted_model.

Outputs, under <runs_dir>/<scenario>/<sim_label>/crispat_gauss[_block<START>_<STEP>]/:
    assigned.parquet          (guide, cell) 0-based int32, column-major
    crispat_batch0/           crispat's assignments, gRNA_thresholds, gRNA_losses and
                              estimated_parameters CSVs; its plot folders are deleted
    crispat_assignments.csv   crispat's combined assignments (full runs only)
    run.json                  status, timings, versions and settings
`seconds` times the ga_gauss call alone, which reads the h5ad itself; `load_seconds` is
reading the parquet and writing that h5ad, which is deleted afterwards. A run whose run.json
says "ok" is skipped; a failed one records status "error" and runs again next time.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import anndata as ad
import crispat
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy import sparse

CRISPAT_COMMIT = "bd2efa7039cbc307b5f7a62a1ffd7a063d1894a3"
METHOD_ID = "crispat_gauss"
# crispat appends "batch<b>/" and "assignments.csv" to the prefix it is given.
CRISPAT_PREFIX = "crispat_"
CRISPAT_BATCH_DIR = CRISPAT_PREFIX + "batch0"
H5AD_NAME = "input.h5ad"
DISTRIBUTIONS = (
    "crispat",
    "torch",
    "pyro-ppl",
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "anndata",
    "scanpy",
    "pyarrow",
)
# Read from crispat/gauss.py at CRISPAT_COMMIT, so a run.json can be read without it.
CRISPAT_FIXED = {
    "transform": "log10(count + 1); with nonzero=True only cells with a nonzero count are fitted",
    "fit_gate": "a guide is fitted only with >= 2 nonzero cells and a maximum count >= 2",
    "model": "two-component Gaussian mixture with one shared scale, MAP by SVI (AutoDelta)",
    "optimizer": "pyro Adam, lr 0.01",
    "elbo": "TraceEnum_ELBO(num_particles=1, max_plate_nesting=1)",
    "initialization": "pyro seeds 0 to 9, the lowest initial loss is trained; "
    "the seed argument 2024 is overwritten",
    "threshold": "smallest count in 1..max at which the higher-mean component has the larger "
    "weighted density; cells at or above it are assigned",
}


def stamp(*parts: object) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] " + "".join(map(str, parts)), flush=True)


def refuse_unless_ignored(out_dir: Path) -> None:
    """Exit unless `out_dir` is outside any git work tree or ignored by git.

    The Python twin of `refuse_unless_ignored()` in eval_common.R: parquet and JSON are not
    covered by a .gitignore pattern, so the directory must be ignored as a whole.
    """
    git = shutil.which("git")
    if git is None:
        return
    out_dir = Path(os.path.abspath(out_dir))
    anchor = out_dir
    while not anchor.is_dir():
        anchor = anchor.parent
    inside = subprocess.run(
        [git, "-C", str(anchor), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
    )
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return
    probe = out_dir / "probe" / "meta.json"
    ignored = subprocess.run(
        [git, "-C", str(anchor), "check-ignore", "-q", str(probe)], capture_output=True
    )
    if ignored.returncode != 0:
        sys.exit(f"refusing to write to {out_dir}: it is inside a git work tree and not ignored")


def installed_crispat() -> dict:
    """The installed crispat's version, URL and commit, from its PEP 610 direct_url.json."""
    dist = importlib.metadata.distribution("crispat")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    return {
        "version": dist.version,
        "url": direct.get("url"),
        "commit": direct.get("vcs_info", {}).get("commit_id"),
    }


def _no_plot_loss(losses, gRNA, output_dir) -> None:
    return None


def patch_plot_loss() -> dict:
    """Replace crispat.gauss.plot_loss, which fit_GMM looks up in its module, with a no-op."""
    gauss = importlib.import_module("crispat.gauss")
    if not callable(getattr(gauss, "plot_loss", None)):
        raise RuntimeError("crispat.gauss has no plot_loss to patch")
    if crispat.ga_gauss is not gauss.ga_gauss:
        raise RuntimeError("crispat.ga_gauss is not crispat.gauss.ga_gauss")
    gauss.plot_loss = _no_plot_loss
    return {"crispat.gauss.plot_loss": "plot_loss patched to a no-op"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic(path: Path, writer) -> None:
    tmp = path.with_name(path.name + ".tmp")
    writer(tmp)
    os.replace(tmp, path)


def write_json_atomic(record: dict, path: Path) -> None:
    text = json.dumps(record, indent=2, allow_nan=False) + "\n"
    write_atomic(path, lambda tmp: tmp.write_text(text))


def run_is_ok(run_json: Path) -> bool:
    if not run_json.exists():
        return False
    try:
        return json.loads(run_json.read_text()).get("status") == "ok"
    except (OSError, ValueError):
        return False


def load_counts(data_dir: Path) -> tuple[sparse.csc_matrix, dict]:
    """The dataset's counts as a cells x guides CSC matrix of int64, and its meta.json."""
    meta = json.loads((data_dir / "meta.json").read_text())
    n_guides, n_cells = int(meta["n_guides"]), int(meta["n_cells"])
    table = pq.read_table(data_dir / "counts.parquet", columns=["guide", "cell", "value"])
    guide = table["guide"].to_numpy()
    cell = table["cell"].to_numpy()
    value = table["value"].to_numpy()
    if len(value) != int(meta["nnz_counts"]):
        raise ValueError(
            f"{len(value)} counts in counts.parquet, meta.json says {meta['nnz_counts']}"
        )
    if len(value) and (value.min() < 0 or guide.min() < 0 or cell.min() < 0):
        raise ValueError("negative count or index in counts.parquet")
    if len(value) and (guide.max() >= n_guides or cell.max() >= n_cells):
        raise ValueError("index out of range in counts.parquet")
    x = sparse.csc_matrix((value.astype(np.int64), (cell, guide)), shape=(n_cells, n_guides))
    if x.nnz != len(value):
        raise ValueError("duplicate (guide, cell) entries in counts.parquet")
    return x, meta


def build_anndata(x: sparse.csc_matrix) -> ad.AnnData:
    """What bin/convert_to_anndata.R built: names as in the R objects, one batch, 0."""
    n_cells, n_guides = x.shape
    adata = ad.AnnData(x)
    adata.obs_names = [f"cell_{j}" for j in range(1, n_cells + 1)]
    adata.var_names = [f"feature_{k}" for k in range(1, n_guides + 1)]
    adata.obs["batch"] = np.zeros(n_cells, dtype=np.int32)
    return adata


def guides_passing_fit_gate(x: sparse.csc_matrix, lo: int, hi: int) -> int:
    """How many of guides lo..hi-1 crispat fits: >= 2 nonzero cells and a maximum >= 2."""
    block = x[:, lo:hi].tocoo()
    keep = block.data > 0
    n_nonzero = np.bincount(block.col[keep], minlength=hi - lo)
    largest = np.zeros(hi - lo, dtype=np.int64)
    np.maximum.at(largest, block.col[keep], block.data[keep])
    return int(np.count_nonzero((n_nonzero >= 2) & (largest >= 2)))


def read_assignments(
    csv_path: Path, x: sparse.csc_matrix, lo: int, hi: int
) -> tuple[np.ndarray, np.ndarray]:
    """crispat's assignments as 0-based (guide, cell) int32, sorted by cell, then guide."""
    n_cells, n_guides = x.shape
    frame = pd.read_csv(csv_path)
    if len(frame) == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    cell_names = pd.Index([f"cell_{j}" for j in range(1, n_cells + 1)])
    guide_names = pd.Index([f"feature_{k}" for k in range(1, n_guides + 1)])
    cell = cell_names.get_indexer(frame["cell"].astype(str))
    guide = guide_names.get_indexer(frame["gRNA"].astype(str))
    if (cell < 0).any() or (guide < 0).any():
        raise ValueError(f"names in {csv_path} that are not in the dataset")
    if guide.min() < lo or guide.max() >= hi:
        raise ValueError(f"guides outside {lo}..{hi - 1} in {csv_path}")
    if len(np.unique(guide.astype(np.int64) * n_cells + cell)) != len(frame):
        raise ValueError(f"duplicate (guide, cell) pairs in {csv_path}")
    counts = np.asarray(x[cell, guide]).ravel()
    if (counts <= 0).any():
        raise ValueError(f"an assignment with a zero count in {csv_path}")
    if "UMI_counts" in frame and not np.array_equal(counts, frame["UMI_counts"].to_numpy()):
        raise ValueError(f"UMI_counts in {csv_path} disagree with counts.parquet")
    order = np.lexsort((guide, cell))
    return guide[order].astype(np.int32), cell[order].astype(np.int32)


def write_assigned(path: Path, guide: np.ndarray, cell: np.ndarray) -> None:
    table = pa.table({"guide": pa.array(guide, pa.int32()), "cell": pa.array(cell, pa.int32())})
    write_atomic(path, lambda tmp: pq.write_table(table, tmp))


def remove_crispat_outputs(out_dir: Path, keep_csvs: bool) -> None:
    """Delete the plot folders and the h5ad, and with keep_csvs=False everything crispat wrote."""
    batch_dir = out_dir / CRISPAT_BATCH_DIR
    for name in ("loss_plots", "fitted_model_plots"):
        shutil.rmtree(batch_dir / name, ignore_errors=True)
    (out_dir / H5AD_NAME).unlink(missing_ok=True)
    if not keep_csvs:
        shutil.rmtree(batch_dir, ignore_errors=True)
        (out_dir / (CRISPAT_PREFIX + "assignments.csv")).unlink(missing_ok=True)
        (out_dir / "assigned.parquet").unlink(missing_ok=True)


def versions() -> dict:
    out = {name: importlib.metadata.version(name) for name in DISTRIBUTIONS}
    out["python"] = platform.python_version()
    out["platform"] = platform.platform()
    return out


def settings(kwargs: dict, patches: dict) -> dict:
    signature = inspect.signature(crispat.ga_gauss).parameters
    effective = {
        name: kwargs.get(name, parameter.default)
        for name, parameter in signature.items()
        if name not in ("input_file", "output_dir")
    }
    return {
        "call": "crispat.ga_gauss(h5ad, out_prefix, **kwargs)",
        "kwargs": kwargs,
        "effective_arguments": effective,
        "crispat_fixed": CRISPAT_FIXED,
        "out_prefix": CRISPAT_PREFIX,
        "patches": patches,
        "plots": "plot folders deleted after the run; plot_fitted_model is disabled in the fork",
        "input": {
            "x_format": "csc",
            "x_dtype": "int64",
            "orientation": "cells x guides",
            "obs_batch": "int32, all 0",
            "built_like": "fishash_analysis bin/convert_to_anndata.R at 192008d",
        },
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }


def run_one(
    data_dir: Path,
    out_dir: Path,
    scenario: str,
    block: tuple[int, int] | None,
    context: dict,
) -> bool:
    run_json = out_dir / "run.json"
    if run_is_ok(run_json):
        stamp("skip ", out_dir, ": run.json says ok")
        return True
    if not (data_dir / "meta.json").exists():
        stamp("skip ", data_dir, ": not exported yet (no meta.json)")
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    remove_crispat_outputs(out_dir, keep_csvs=False)
    h5ad = out_dir / H5AD_NAME
    kwargs: dict = {"nonzero": True}
    if block is not None:
        kwargs.update(start_gRNA=block[0], step=block[1])
    record: dict = {
        "status": "error",
        "method_id": out_dir.name,
        "scenario": scenario,
        "sim_label": data_dir.name,
        "block": None if block is None else list(block),
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    try:
        start = time.perf_counter()
        x, meta = load_counts(data_dir)
        n_cells, n_guides = x.shape
        lo = 0 if block is None else block[0]
        hi = n_guides if block is None else min(block[0] + block[1], n_guides)
        if lo >= hi:
            raise ValueError(f"block starts at guide {lo}, the dataset has {n_guides}")
        build_anndata(x).write_h5ad(h5ad)
        load_seconds = time.perf_counter() - start

        start = time.perf_counter()
        crispat.ga_gauss(str(h5ad), str(out_dir / CRISPAT_PREFIX), **kwargs)
        seconds = time.perf_counter() - start

        # ga_gauss writes the combined file only when step is None.
        if block is None:
            csv_name = CRISPAT_PREFIX + "assignments.csv"
        else:
            csv_name = CRISPAT_BATCH_DIR + "/assignments.csv"
        guide, cell = read_assignments(out_dir / csv_name, x, lo, hi)
        write_assigned(out_dir / "assigned.parquet", guide, cell)
        record.update(
            status="ok",
            seconds=seconds,
            load_seconds=load_seconds,
            n_assigned=int(len(guide)),
            n_guides_assigned=int(len(np.unique(guide))),
            n_guides_covered=hi - lo,
            guide_range=[lo, hi],
            n_guides_fit_gate=guides_passing_fit_gate(x, lo, hi),
            seconds_per_guide=seconds / (hi - lo),
            assignments_csv=csv_name,
            dataset={
                "n_guides": n_guides,
                "n_cells": n_cells,
                "nnz_counts": int(x.nnz),
                "counts_sha256": sha256_file(data_dir / "counts.parquet"),
                "rds_sha256": meta.get("rds_sha256"),
                "upstream_sha": meta.get("upstream_sha"),
            },
        )
    except Exception as error:  # recorded, and the next dataset still runs
        record.update(error=repr(error), traceback=traceback.format_exc())
        stamp("error in ", out_dir, ": ", repr(error))
    finally:
        remove_crispat_outputs(out_dir, keep_csvs=True)
    record.update(
        finished_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        crispat=context["crispat"],
        versions=context["versions"],
        settings=settings(kwargs, context["patches"]),
    )
    write_json_atomic(record, run_json)
    if record["status"] == "ok":
        stamp(
            "done ",
            out_dir,
            f": {record['seconds']:.1f} s for {record['n_guides_covered']} guides",
            f" ({record['seconds_per_guide']:.2f} s/guide), {record['n_assigned']} assigned",
        )
    return record["status"] == "ok"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run crispat ga_gauss on simulated datasets.",
        usage="%(prog)s <sims_dir> <scenario> <runs_dir> [--block START STEP] [sim_label ...]",
    )
    parser.add_argument("sims_dir", type=Path)
    parser.add_argument("scenario")
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("sim_labels", nargs="*", metavar="sim_label")
    parser.add_argument("--block", nargs=2, type=int, metavar=("START", "STEP"))
    args = parser.parse_intermixed_args(argv)
    if args.block is not None and (args.block[0] < 0 or args.block[1] < 1):
        parser.error("--block needs START >= 0 and STEP >= 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    refuse_unless_ignored(args.runs_dir)
    installed = installed_crispat()
    if installed["commit"] != CRISPAT_COMMIT:
        sys.exit(f"crispat is at {installed['commit']}, this script needs {CRISPAT_COMMIT}")

    scenario_dir = args.sims_dir / args.scenario
    manifest = json.loads((scenario_dir / "manifest.json").read_text())
    if manifest.get("status") != "ok":
        sys.exit(f"{scenario_dir / 'manifest.json'} does not say ok")
    labels = [d["sim_label"] for d in manifest["datasets"]]
    if args.sim_labels:
        unknown = sorted(set(args.sim_labels) - set(labels))
        if unknown:
            sys.exit(f"not in the {args.scenario} manifest: {', '.join(unknown)}")
        labels = args.sim_labels

    torch.set_num_threads(1)
    context = {"crispat": installed, "versions": versions(), "patches": patch_plot_loss()}
    block = None if args.block is None else (args.block[0], args.block[1])
    method_id = METHOD_ID if block is None else f"{METHOD_ID}_block{block[0]}_{block[1]}"
    failed = []
    for label in labels:
        out_dir = args.runs_dir / args.scenario / label / method_id
        if not run_one(scenario_dir / label, out_dir, args.scenario, block, context):
            failed.append(label)
    stamp("done ", args.scenario, ": ", len(labels) - len(failed), " ok, ", len(failed), " failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
