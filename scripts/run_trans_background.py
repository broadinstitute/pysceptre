"""Run a trans background in shards and the specificity check on it.

Inputs come from `build_trans_pairs.py` (DIR = its output directory). Every
step runs in a batch job; every path comes from the command line.

  run_trans_background.py fit-genes --dataset H5MU --task I --n-tasks K --out FITS
  run_trans_background.py shard --dataset H5MU --pairs-dir DIR --shard I --gene-fits FITS
      --reference-tsv CIS.tsv --out SHARDS
  run_trans_background.py combine --pairs-dir DIR --shards SHARDS --out COMBINED
  run_trans_background.py specificity --cis-tsv CIS.tsv --pairs-dir DIR --trans COMBINED --out OUT

With `--backed`, `fit-genes` and `shard` read each gene from the dataset as
they reach it instead of holding the counts in memory; the dataset must hold
only its cells in use (`write_in_use_h5mu.py`). `fit-genes` computes width-1
gene fits for one subset of the cis genes with
`pysceptre.pipeline.gene_cache.fit_gene_cache`, the engine's own call. `shard`
tests one shard's pairs with `pysceptre.pipeline.trans.iter_trans_analysis`:
the shard's reference (cis) pairs in full, compared bit for bit with the
reference result as they come (any difference stops the shard), an audit
sample both screened and in full, and the rest screened. Results are written
as parquet parts per group of genes, each renamed into place when complete,
so a restarted shard skips finished parts. `combine` checks that every
QC-passing trans pair appears exactly once and that the audit sample agrees
with the screens, and writes one table. `specificity` runs
`run_specificity_check` at `--cutoff`, or by default at the BH cutoff at
`--bh-alpha` over the finite cis p-values left after `--exclude-pairs`: the
cutoff the cis analysis was called at, when `--exclude-pairs` flags the pairs
that analysis left out of its correction. Cis pairs whose target or gene has
no position, or whose element and gene lie on different chromosomes, are then
dropped from the check without moving the cutoff, and listed in `summary.json`.
See docs/design.md, "The trans background".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from sceptre_io import load_export_compact, load_h5mu  # noqa: E402

from pysceptre.analytical_power.inputs import bh_nominal_cutoff  # noqa: E402
from pysceptre.pipeline.discovery import _CI_Z, _as_pct, chunk_layout_hash  # noqa: E402
from pysceptre.pipeline.fingerprint import environment_fingerprint  # noqa: E402
from pysceptre.pipeline.gene_cache import (  # noqa: E402
    fit_gene_cache,
    load_gene_cache,
    save_gene_cache,
)
from pysceptre.pipeline.trans import (  # noqa: E402
    SCREEN_CODES,
    iter_trans_analysis,
    trans_fit_chunks,
)
from pysceptre.specificity import run_specificity_check  # noqa: E402

_SIDES = {-1: "left", 0: "both", 1: "right"}
_RESULT = ("p_value", "fold_change", "se_fold_change", "z_orig", "stage")
_AUDIT = tuple(f"audit_{c}" for c in _RESULT)


def _tmp(path: Path) -> Path:
    """A temporary sibling unique to this job attempt, renamed into place when complete."""
    job = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    restart = os.environ.get("SLURM_RESTART_COUNT", "0")
    return path.with_name(f".{path.name}.tmp.{job}.{restart}")


def _write_parquet(df: pd.DataFrame, path: Path) -> str:
    tmp = _tmp(path)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    os.replace(tmp, path)
    return digest


def _write_json(obj, path: Path) -> None:
    tmp = _tmp(path)
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """A fixed, well-mixed hash of int64 keys, for choosing the audit sample."""
    z = x.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def _layout(pairs_dir: Path):
    layout = json.loads((pairs_dir / "layout.json").read_text())
    return layout, (layout["fit_chunk_width"], layout["n_fit_chunks"])


def _manifest(a: argparse.Namespace, pairs_dir: Path) -> dict:
    path = Path(a.manifest) if getattr(a, "manifest", None) else pairs_dir / "shard_manifest.json"
    return json.loads(path.read_text())


def _reference_chunks(ex, layout, expected):
    reference = ex.pairs[["response_id", "grna_target"]].reset_index(drop=True)
    chunks = trans_fit_chunks(
        ex.grna_target_cells,
        reference,
        ex.covariate_matrix.shape[0],
        target_chunk_size=layout["target_chunk_size"],
        chunk_memory_gb=layout["chunk_memory_gb"],
        expected_fit_chunks=expected,
    )
    if chunk_layout_hash(chunks) != layout["layout_hash"]:
        raise SystemExit("the dataset's reference layout differs from layout.json")
    return reference, chunks


def _load(a: argparse.Namespace):
    """The dataset: the reference run's genes held in memory, or with `--backed` every gene
    read from the file as needed (a file holding only its cells in use; see
    `write_in_use_h5mu.py`)."""
    if getattr(a, "backed", False):
        return load_h5mu(a.dataset, backed=True)
    return load_export_compact(a.dataset, genes="pairs")


def _reference_genes(ex) -> list[str]:
    """The genes of `ex`'s pairs in matrix order, whether `ex` holds only those or every gene."""
    wanted = set(ex.pairs["response_id"])
    return [g for g in ex.gene_ids if g in wanted]


def cmd_fit_genes(a: argparse.Namespace) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    whole = out / f"gene_fits_{a.task:03d}of{a.n_tasks:03d}.npz"
    # A file from before the fingerprint recorded the covariate layout is refused by every
    # load; stop rather than skip it as done.
    for old in sorted(out.glob(f"gene_fits_{a.task:03d}of{a.n_tasks:03d}*.npz")):
        try:
            load_gene_cache(old, check_environment=False)
        except ValueError as e:
            raise SystemExit(f"{old} cannot be reused ({e}); delete it and rerun fit-genes") from e
    if whole.exists():
        print(f"exists: {whole}")
        return
    t0 = time.time()
    # The reference run's genes only, in matrix order: the list build_trans_pairs.py writes.
    ex = _load(a)
    genes = _reference_genes(ex)
    mine = [str(g) for g in np.array_split(np.asarray(genes, dtype=object), a.n_tasks)[a.task]]
    print(
        f"loaded in {time.time() - t0:.0f}s; fitting {len(mine)} of {len(genes)} genes", flush=True
    )
    # Parts written as they finish, so a preempted task resumes where it stopped.
    n_parts = max(1, -(-len(mine) // a.part_size))
    for i, part in enumerate(np.array_split(np.asarray(mine, dtype=object), n_parts)):
        path = out / f"gene_fits_{a.task:03d}of{a.n_tasks:03d}_part{i:03d}.npz"
        if path.exists():
            continue
        t0 = time.time()
        cache = fit_gene_cache(
            ex.response_matrix,
            ex.gene_ids,
            ex.covariate_matrix,
            genes=[str(g) for g in part],
            n_jobs=a.n_jobs,
            glm_solver=a.glm_solver,
        )
        seconds = time.time() - t0
        tmp = _tmp(path).with_suffix(".npz")
        save_gene_cache(cache, tmp)
        os.replace(tmp, path)
        _write_json(
            {"genes": len(part), "seconds": seconds, "host": socket.gethostname()},
            path.with_suffix(".done.json"),
        )
        print(f"part {i}: fit {len(part)} genes in {seconds:.0f}s -> {path}", flush=True)


def _reference_table(tsv: Path) -> pd.DataFrame:
    ref = pd.read_csv(tsv, sep="\t", float_precision="round_trip")
    return ref.set_index(["response_id", "grna_target"])


def _check_reference(rows: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    """Rows of `rows` (full results) whose values differ in any bit from the reference result."""
    exp = expected.loc[list(zip(rows["response_id"], rows["grna_target"], strict=True))]
    fc, se = rows["fold_change"].to_numpy(), rows["se_fold_change"].to_numpy()
    got = {c: rows[c].to_numpy(float) for c in _RESULT}
    got["pct_change_es"] = _as_pct(fc)
    got["pct_change_es_ci_low"] = _as_pct(fc - _CI_Z * se)
    got["pct_change_es_ci_high"] = _as_pct(fc + _CI_Z * se)
    bad = np.zeros(len(rows), dtype=bool)
    for c, values in got.items():
        bad |= ~_equal_nan(values, exp[c].to_numpy(float))
    return rows.loc[bad]


def _equal_nan(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (x == y) | (np.isnan(x) & np.isnan(y))


def cmd_shard(a: argparse.Namespace) -> None:
    pairs_dir = Path(a.pairs_dir)
    out = Path(a.out) / f"shard_{a.shard:04d}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE.json").exists():
        print(f"shard {a.shard} is already done")
        return
    fingerprint = environment_fingerprint()
    layout, expected = _layout(pairs_dir)
    manifest = _manifest(a, pairs_dir)
    if manifest["layout_hash"] != layout["layout_hash"]:
        raise SystemExit("shard_manifest.json was built for a different layout")
    chunk_ids = manifest["shards"][a.shard]

    t0 = time.time()
    # The trans genes are the reference run's genes; nothing else is read.
    ex = _load(a)
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    reference, _ = _reference_chunks(ex, layout, expected)
    meta = ex.metadata
    side = _SIDES[int(meta["side_code"])]
    elements = pd.read_parquet(pairs_dir / "elements.parquet")
    genes = pd.read_parquet(pairs_dir / "genes.parquet")
    n_targets = len(elements)
    in_shard = elements["fit_chunk"].isin(chunk_ids).to_numpy()
    shard_codes = elements.loc[in_shard, "target_code"].to_numpy()

    tp = pq.read_table(
        pairs_dir / "trans_pairs.parquet",
        columns=["gene_code", "target_code"],
        filters=[("pass_qc", "==", True), ("target_code", "in", shard_codes.tolist())],
    )
    trans_key = tp["gene_code"].to_numpy().astype(np.int64) * n_targets + tp[
        "target_code"
    ].to_numpy().astype(np.int64)
    t_code = dict(zip(elements["grna_target"], elements["target_code"], strict=True))
    g_code = dict(zip(genes["response_id"], genes["gene_code"], strict=True))
    shard_targets = set(elements.loc[in_shard, "grna_target"])
    ref = reference.loc[reference["grna_target"].isin(shard_targets)]
    ref_key = ref["response_id"].map(g_code).to_numpy(np.int64) * n_targets + ref[
        "grna_target"
    ].map(t_code).to_numpy(np.int64)

    key = np.union1d(trans_key, ref_key)  # sorted: gene code major, so gene_ids order
    is_ref = np.isin(key, ref_key)
    is_trans = np.isin(key, trans_key)
    audit = is_trans & ~is_ref & (_splitmix64(key) % np.uint64(a.audit_every) == 0)
    mode = np.where(is_ref, "full", np.where(audit, "audit", "screened"))
    gcode, tcode = key // n_targets, key % n_targets

    shard_genes = np.unique(gcode)
    part_of_gene = dict(
        zip(shard_genes, np.arange(shard_genes.size) // a.genes_per_part, strict=True)
    )
    part = np.fromiter((part_of_gene[g] for g in gcode), dtype=np.int64, count=gcode.size)
    done = {int(p.name.split("_")[1].split(".")[0]) for p in out.glob("part_*.done.json")}
    todo = ~np.isin(part, list(done))
    print(
        f"shard {a.shard}: {len(chunk_ids)} chunks, {shard_codes.size} targets, "
        f"{int(is_trans.sum()):,} trans pairs ({int(audit.sum()):,} audited), "
        f"{int(is_ref.sum()):,} reference pairs, {len(done)} part(s) already done",
        flush=True,
    )

    cache = load_gene_cache(
        sorted(Path(a.gene_fits).glob("gene_fits_*.npz")),
        response_matrix=ex.response_matrix,
        gene_ids=ex.gene_ids,
        covariate_matrix=ex.covariate_matrix,
        glm_solver=a.glm_solver,
    )
    expected_ref = _reference_table(Path(a.reference_tsv))
    work = pd.DataFrame(
        {
            "response_id": genes["response_id"].to_numpy()[gcode[todo]],
            "grna_target": elements["grna_target"].to_numpy()[tcode[todo]],
            "mode": mode[todo],
        }
    )
    w_gcode, w_tcode, w_part = gcode[todo], tcode[todo], part[todo]
    w_ref, w_trans, w_audit = is_ref[todo], is_trans[todo], audit[todo]

    stats = {"reference_checked": 0, "rows_written": 0, "screen": dict.fromkeys(SCREEN_CODES, 0)}
    buffer: list = []
    current = None
    t0 = time.time()

    def flush(p: int) -> None:
        rows = np.concatenate([b[0] for b in buffer])
        cols = {c: np.concatenate([b[1][c] for b in buffer]) for c in buffer[0][1]}
        base = pd.DataFrame(
            {
                "gene_code": w_gcode[rows].astype(np.int32),
                "target_code": w_tcode[rows].astype(np.int32),
            }
        )
        for c in _RESULT + ("p_value_floor",):
            base[c] = cols[c]
        base["screen"] = cols["screen"]
        base["n_null_evaluated"] = cols["n_null_evaluated"]
        trans_rows, ref_rows, audit_rows = w_trans[rows], w_ref[rows], w_audit[rows]
        digests = {"part": _write_parquet(base.loc[trans_rows], out / f"part_{p:05d}.parquet")}
        if ref_rows.any():
            digests["reference"] = _write_parquet(
                base.loc[ref_rows], out / f"reference_{p:05d}.parquet"
            )
        if audit_rows.any():
            aud = base.loc[audit_rows].copy()
            for c in _AUDIT:
                aud[c] = cols[c][audit_rows]
            digests["audit"] = _write_parquet(aud, out / f"audit_{p:05d}.parquet")
        for code, name in enumerate(SCREEN_CODES):
            stats["screen"][name] += int((cols["screen"][trans_rows] == code).sum())
        stats["rows_written"] += int(trans_rows.sum())
        _write_json(
            {
                "part": p,
                "trans_rows": int(trans_rows.sum()),
                "reference_rows": int(ref_rows.sum()),
                "audit_rows": int(audit_rows.sum()),
                "sha256": digests,
                "seconds_since_start": time.time() - t0,
            },
            out / f"part_{p:05d}.done.json",
        )
        print(
            f"part {p} written: {int(trans_rows.sum()):,} trans rows, {time.time() - t0:.0f}s",
            flush=True,
        )
        buffer.clear()

    for r in iter_trans_analysis(
        ex.response_matrix,
        ex.gene_ids,
        ex.covariate_matrix,
        ex.grna_target_cells,
        work,
        reference_pairs=reference,
        seed=a.seed,
        side=side,
        resampling_approximation=meta["resampling_approximation"],
        target_chunk_size=layout["target_chunk_size"],
        chunk_memory_gb=layout["chunk_memory_gb"],
        expected_fit_chunks=expected,
        fit_chunk_ids=chunk_ids,
        # Checked against the data above, once; a plain dict is not checked again.
        gene_fits=cache.fits,
        p_exact_below=a.p_exact_below,
        block_rows=a.block_rows,
        n_jobs=a.n_jobs,
        backend=a.backend,
        fit_concurrency=a.fit_concurrency,
        glm_solver=a.glm_solver,
    ):
        p = int(w_part[r.rows[0]])
        if current is not None and p != current:
            flush(current)
        current = p
        refs = w_ref[r.rows]
        if refs.any():
            rows = work.iloc[r.rows[refs]][["response_id", "grna_target"]].reset_index(drop=True)
            for c in _RESULT:
                rows[c] = r.columns[c][refs]
            bad = _check_reference(rows, expected_ref)
            stats["reference_checked"] += int(refs.sum())
            if len(bad):
                bad.to_csv(out / "reference_mismatch.tsv", sep="\t", index=False)
                raise SystemExit(
                    f"{len(bad)} reference pair(s) differ from {a.reference_tsv}; see "
                    f"{out / 'reference_mismatch.tsv'}. Wrong node type or environment?"
                )
        buffer.append((r.rows, r.columns))
    if buffer:
        flush(current)
    _write_json(
        {
            "shard": a.shard,
            "chunks": chunk_ids,
            "seconds": time.time() - t0,
            "host": socket.gethostname(),
            "fingerprint": fingerprint,
            "glm_solver": a.glm_solver,
            **stats,
        },
        out / "DONE.json",
    )
    print(f"shard {a.shard} done in {time.time() - t0:.0f}s: {stats}")


def cmd_combine(a: argparse.Namespace) -> None:
    pairs_dir, shards_dir, out = Path(a.pairs_dir), Path(a.shards), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    layout, _ = _layout(pairs_dir)
    manifest = _manifest(a, pairs_dir)
    n_targets = layout["n_elements"]
    missing = [
        i
        for i in range(len(manifest["shards"]))
        if not (shards_dir / f"shard_{i:04d}" / "DONE.json").exists()
    ]
    if missing and not a.allow_missing:
        raise SystemExit(
            f"shards not done: {missing}; resubmit with --array={','.join(map(str, missing))}"
        )
    done = [i for i in range(len(manifest["shards"])) if i not in missing]
    parts = sorted(p for i in done for p in (shards_dir / f"shard_{i:04d}").glob("part_*.parquet"))
    table = pq.ParquetDataset(parts).read().to_pandas() if parts else pd.DataFrame()
    key = table["gene_code"].to_numpy(np.int64) * n_targets + table["target_code"].to_numpy(
        np.int64
    )
    order = np.argsort(key, kind="stable")
    table, key = table.iloc[order].reset_index(drop=True), key[order]
    codes = pq.read_table(
        pairs_dir / "trans_pairs.parquet", columns=["gene_code", "target_code", "pass_qc"]
    ).to_pandas()
    elements = pd.read_parquet(pairs_dir / "elements.parquet")
    chunk_of = elements["fit_chunk"].to_numpy()
    covered = np.isin(
        chunk_of[codes["target_code"].to_numpy()], [c for i in done for c in manifest["shards"][i]]
    )
    want = codes.loc[codes["pass_qc"].to_numpy() & covered]
    want_key = np.sort(
        want["gene_code"].to_numpy(np.int64) * n_targets + want["target_code"].to_numpy(np.int64)
    )
    duplicated = int((np.diff(key) == 0).sum())
    exact = duplicated == 0 and np.array_equal(key, want_key)
    screen = table["screen"].to_numpy()
    p = table["p_value"].to_numpy()
    nan_tested = int((np.isnan(p) & (screen == 0) & np.isfinite(table["z_orig"].to_numpy())).sum())
    audits = [p for i in done for p in (shards_dir / f"shard_{i:04d}").glob("audit_*.parquet")]
    audit = pq.ParquetDataset(audits).read().to_pandas() if audits else pd.DataFrame()
    audit_checks = {}
    if len(audit):
        none = audit["screen"] == 0
        early = audit["screen"] == 2
        fcs = audit["screen"] == 1
        same = np.ones(len(audit), dtype=bool)
        for c in _RESULT:
            same &= _equal_nan(audit[c].to_numpy(float), audit[f"audit_{c}"].to_numpy(float))
        audit_checks = {
            "rows": int(len(audit)),
            "none_not_equal_to_full": int((none & ~same).sum()),
            "early_stop_full_p_below_floor": int(
                (early & (audit["audit_p_value"] < audit["p_value_floor"])).sum()
            ),
            "fold_change_screened_with_fc_below_1": int(
                (fcs & (audit["audit_fold_change"] < 1)).sum()
            ),
            "full_stage1_p_quantiles": np.nanquantile(
                audit.loc[audit["audit_stage"] == 1, "audit_p_value"], [0.05, 0.25, 0.5, 0.75, 0.95]
            ).tolist(),
            "full_share_p_le_0_05": float((audit["audit_p_value"] <= 0.05).mean()),
        }
    summary = {
        "shards_done": done,
        "shards_missing": missing,
        "rows": int(len(table)),
        "expected_rows": int(want_key.size),
        "exactly_once": bool(exact),
        "duplicated_keys": duplicated,
        "nan_p_on_unscreened_finite_z": nan_tested,
        "screen_counts": {n: int((screen == c).sum()) for c, n in enumerate(SCREEN_CODES)},
        "stage_counts": table["stage"].value_counts(dropna=False).to_dict(),
        "n_null_evaluated_total": int(table["n_null_evaluated"].sum()),
        "audit": audit_checks,
        "hosts": [
            json.loads((shards_dir / f"shard_{i:04d}" / "DONE.json").read_text())["host"]
            for i in done
        ],
        "reference_checked": sum(
            json.loads((shards_dir / f"shard_{i:04d}" / "DONE.json").read_text())[
                "reference_checked"
            ]
            for i in done
        ),
    }
    _write_json(summary, out / "summary.json")
    print(json.dumps(summary, indent=1, default=str))
    bad = (
        not exact
        or nan_tested
        or any(
            audit_checks.get(k, 0)
            for k in (
                "none_not_equal_to_full",
                "early_stop_full_p_below_floor",
                "fold_change_screened_with_fc_below_1",
            )
        )
    )
    if bad:
        # An earlier run's table must not outlive the summary that now says it fails.
        (out / "trans_result.parquet").unlink(missing_ok=True)
        raise SystemExit(
            "combine checks failed, trans_result.parquet not written; see summary.json"
        )
    _write_parquet(table, out / "trans_result.parquet")


def cmd_specificity(a: argparse.Namespace) -> None:
    pairs_dir, out = Path(a.pairs_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    elements = pd.read_parquet(pairs_dir / "elements.parquet")
    genes = pd.read_parquet(pairs_dir / "genes.parquet")
    measured = pd.read_parquet(pairs_dir / "measured_genes.parquet")
    cis = pd.read_csv(a.cis_tsv, sep="\t", float_precision="round_trip")
    notes = {"cis_pairs": int(len(cis))}
    if a.exclude_pairs:
        excl = pd.read_csv(
            a.exclude_pairs, sep="\t", usecols=["response_id", "grna_target", a.exclude_column]
        )
        excl = excl.loc[excl[a.exclude_column].astype(bool), ["response_id", "grna_target"]]
        drop = pd.MultiIndex.from_frame(cis[["response_id", "grna_target"]]).isin(
            pd.MultiIndex.from_frame(excl)
        )
        notes["excluded_by_" + a.exclude_column] = int(drop.sum())
        cis = cis.loc[~drop].reset_index(drop=True)
    # The cutoff is taken here: the pairs dropped below for their positions do not move it.
    if a.cutoff is not None:
        cutoff = a.cutoff
    else:
        p = cis["p_value"].to_numpy(float)
        cutoff = bh_nominal_cutoff(p, a.bh_alpha)
        notes["cutoff_bh"] = {"alpha": a.bh_alpha, "p_values": int(np.isfinite(p).sum())}
    notes["cutoff"] = cutoff
    chrom = dict(zip(measured["response_id"], measured["chrom"], strict=True))
    e_chrom = dict(zip(elements["grna_target"], elements["chrom"], strict=True))
    e_of, g_of = cis["grna_target"].map(e_chrom), cis["response_id"].map(chrom)
    unplaced = (e_of.isna() | g_of.isna()).to_numpy()
    spans = ~unplaced & (e_of != g_of).to_numpy()
    for key, rows in (
        ("cis_pairs_without_position_dropped", unplaced),
        ("cis_pairs_spanning_chromosomes_dropped", spans),
    ):
        notes[key] = cis.loc[rows, ["grna_target", "response_id"]].values.tolist()
    cis = cis.loc[~(unplaced | spans)].reset_index(drop=True)

    t = pq.read_table(a.trans).to_pandas()
    if a.subset_shards:
        manifest = _manifest(a, pairs_dir)
        keep_chunks = [c for i in a.subset_shards for c in manifest["shards"][i]]
        keep = elements.loc[elements["fit_chunk"].isin(keep_chunks), "target_code"].to_numpy()
        t = t.loc[t["target_code"].isin(keep)]
        notes["subset_shards"] = a.subset_shards
    trans = pd.DataFrame(
        {
            "response_id": pd.Categorical.from_codes(
                t["gene_code"].to_numpy(), genes["response_id"]
            ),
            "grna_target": pd.Categorical.from_codes(
                t["target_code"].to_numpy(), elements["grna_target"]
            ),
            "p_value": t["p_value"].to_numpy(),
            "fold_change": t["fold_change"].to_numpy(),
            "screen": np.asarray(SCREEN_CODES, dtype=object)[t["screen"].to_numpy()],
            "p_value_floor": t["p_value_floor"].to_numpy(),
        }
    )
    del t
    t0 = time.time()
    res = run_specificity_check(
        cis,
        trans,
        elements[["grna_target", "chrom", "centre"]],
        measured[["response_id", "chrom", "tss"]],
        cutoff,
        measured_genes=set(measured["response_id"]),
        n_bootstrap=a.n_bootstrap,
        seed=a.seed,
    )
    notes["seconds"] = time.time() - t0
    for name in (
        "by_distance",
        "by_distance_without_broad",
        "broad_effect",
        "per_gene",
        "links",
        "tss_targets",
        "gene_lookup",
        "far_links",
    ):
        getattr(res, name).to_csv(out / f"{name}.tsv", sep="\t", index=False)
    bg = res.background
    _write_parquet(
        bg.astype({c: str for c in ("grna_target", "response_id") if c in bg}),
        out / "background.parquet",
    )
    notes["background_rate"] = res.background_rate
    notes["background_tests"] = int(len(bg))
    notes["background_calls"] = int(bg["background"].sum())
    _write_json(notes, out / "summary.json")
    print(res.by_distance.to_string())
    print(
        json.dumps(
            {
                k: len(v) if k.startswith("cis_pairs_") and k.endswith("_dropped") else v
                for k, v in notes.items()
            },
            indent=1,
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit-genes")
    f.add_argument("--dataset", required=True)
    f.add_argument("--task", type=int, required=True)
    f.add_argument("--n-tasks", type=int, required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--n-jobs", type=int, default=1)
    f.add_argument("--part-size", type=int, default=170, help="genes per saved part")
    f.add_argument("--glm-solver", choices=["outer", "structured"], default="outer")
    f.add_argument("--backed", action="store_true", help="read counts from the dataset as needed")
    f.set_defaults(fn=cmd_fit_genes)

    s = sub.add_parser("shard")
    s.add_argument("--dataset", required=True)
    s.add_argument("--pairs-dir", required=True)
    s.add_argument("--shard", type=int, required=True)
    s.add_argument("--gene-fits", required=True, help="directory of gene_fits_*.npz")
    s.add_argument("--reference-tsv", required=True, help="the reference (cis) run's result")
    s.add_argument("--out", required=True)
    s.add_argument("--manifest", help="default: PAIRS_DIR/shard_manifest.json")
    s.add_argument("--seed", type=int, default=0, help="the reference run's seed")
    s.add_argument("--n-jobs", type=int, default=1)
    s.add_argument("--backend", choices=["fork", "thread"])
    s.add_argument(
        "--fit-concurrency",
        type=int,
        default=12,
        help="target chunks fit at once (under --glm-solver outer each is one batched fit, "
        "under structured its targets are fit on a thread pool)",
    )
    s.add_argument("--audit-every", type=int, default=1000, help="audit about 1 pair in this many")
    s.add_argument("--genes-per-part", type=int, default=256)
    s.add_argument("--p-exact-below", type=float, default=0.05)
    s.add_argument("--block-rows", type=int, default=32)
    s.add_argument(
        "--glm-solver",
        choices=["outer", "structured"],
        default="outer",
        help="the reference run's; --gene-fits must hold fits made with it",
    )
    s.add_argument("--backed", action="store_true", help="read counts from the dataset as needed")
    s.set_defaults(fn=cmd_shard)

    c = sub.add_parser("combine")
    c.add_argument("--pairs-dir", required=True)
    c.add_argument("--shards", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--manifest", help="default: PAIRS_DIR/shard_manifest.json")
    c.add_argument("--allow-missing", action="store_true", help="combine the finished shards only")
    c.set_defaults(fn=cmd_combine)

    p = sub.add_parser("specificity")
    p.add_argument("--cis-tsv", required=True)
    p.add_argument("--pairs-dir", required=True)
    p.add_argument("--trans", required=True, help="combine's trans_result.parquet")
    p.add_argument("--out", required=True)
    p.add_argument("--manifest", help="default: PAIRS_DIR/shard_manifest.json")
    p.add_argument(
        "--cutoff",
        type=float,
        help="nominal cutoff; default: BH at --bh-alpha over the finite --cis-tsv p-values "
        "left after --exclude-pairs, before cis pairs are dropped for their positions",
    )
    p.add_argument("--bh-alpha", type=float, default=0.1, help="FDR level of the default cutoff")
    p.add_argument(
        "--exclude-pairs",
        help="TSV with response_id, grna_target and a flag column; flagged cis pairs leave "
        "the check and the default cutoff",
    )
    p.add_argument("--exclude-column", default="self_targeting", help="the flag column")
    p.add_argument("--subset-shards", type=int, nargs="*", help="an interim run on these shards")
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_specificity)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
