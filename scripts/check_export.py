#!/usr/bin/env python3
"""Check an export on real data, which `tests/validation/test_export_units.py` cannot.

That test covers the format contract on a 40-cell fixture and runs in CI. This
covers the properties that only appear at scale and only on a real screen --
the many-to-many gRNA -> target map, the two cell spaces, and the size of what
`--all-cells` changes -- so it needs a dataset and is run by hand after touching
the exporter.

    scripts/check_export.py <export>                    # one export
    scripts/check_export.py <export> --compare <other>  # and against another

An <export> is either a directory holding the R extraction's .parquet
intermediate or a dataset.h5mu. `--compare` is most useful as
`--all-cells export vs. plain export of the same object`: the whole point of the
subsetting loader is that the first, read normally, is the second.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from sceptre_io import load_export, subset_to_cells_in_use  # noqa: E402

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(label)


def guides_by_target(design) -> dict[str, list[str]]:
    """A MULTIMAP, because the map is many-to-many.

    On day0, 1,673 of 43,736 guides sit inside two or three overlapping
    candidate elements. A dict keyed by guide keeps one target each and makes
    216 of 3,071 targets look like they lost guides -- so the check that would
    catch that must not itself contain it.
    """
    out: dict[str, list[str]] = {}
    for grna_id, target in zip(
        design["grna_id"].astype(str), design["grna_target"].astype(str), strict=True
    ):
        out.setdefault(target, []).append(grna_id)
    return out


def check_one(export, label: str) -> None:
    print(f"\n== {label} ==")
    print(f"   {export.describe()}")
    meta = export.metadata
    n_cells = meta["n_cells"]
    in_use = np.asarray(export.in_use, dtype=bool)
    print(
        f"   cells: {n_cells:,} in the file, {int(in_use.sum()):,} passing QC"
        f"{'  (--all-cells)' if not in_use.all() else ''}"
    )

    for kind, cells in (
        ("target unions", export.grna_target_cells),
        ("NTC gRNAs", export.ntc_grna_cells),
        ("targeting gRNAs", export.targeting_grna_cells),
    ):
        if cells is None:
            print(f"   {kind}: absent")
            continue
        check(
            f"{kind}: cells within [0, n_cells)",
            all(c.size == 0 or (c.min() >= 0 and c.max() < n_cells) for c in cells.values()),
            f"{len(cells):,} units",
        )

    if export.targeting_grna_cells is None or export.grna_target_data_frame is None:
        print("   no individual targeting gRNAs; skipping the union round-trip")
        return

    by_target = guides_by_target(export.grna_target_data_frame)
    tg = export.targeting_grna_cells
    mismatch = []
    for target, cells in export.grna_target_cells.items():
        guides = [g for g in by_target.get(target, []) if g in tg]
        union = np.unique(np.concatenate([tg[g] for g in guides])) if guides else np.array([])
        if not np.array_equal(union, np.unique(cells)):
            mismatch.append(target)
    check(
        "a target's cells are the union of its guides' cells",
        not mismatch,
        f"{len(export.grna_target_cells):,} targets, {len(mismatch)} mismatched"
        + (f", e.g. {mismatch[:2]}" if mismatch else ""),
    )

    counts = export.grna_target_data_frame.groupby("grna_id")["grna_target"].nunique()
    multi = set(counts[counts > 1].index.astype(str))
    touched = [t for t, gs in by_target.items() if any(g in multi for g in gs)]
    print(
        f"   {len(multi):,} guides belong to more than one target, "
        f"reaching {len(touched):,} of {len(by_target):,} targets -- these are the targets a "
        f"collapsed map would silently shrink"
    )


def compare(a, b, label_a: str, label_b: str) -> None:
    print(f"\n== {label_a} == {label_b} ==")
    check("gene_ids", a.gene_ids == b.gene_ids)
    check(
        "response_matrix",
        a.response_matrix.shape == b.response_matrix.shape
        and a.response_matrix.nnz == b.response_matrix.nnz
        and not (a.response_matrix != b.response_matrix).nnz,
        f"{a.response_matrix.shape}, {a.response_matrix.nnz:,} nonzeros",
    )
    check(
        "covariate_matrix",
        np.array_equal(a.covariate_matrix, b.covariate_matrix),
        f"{a.covariate_matrix.shape}",
    )
    for attr in ("grna_target_cells", "ntc_grna_cells", "targeting_grna_cells"):
        x, y = getattr(a, attr), getattr(b, attr)
        if x is None or y is None:
            check(attr, x is None and y is None, "absent from one side")
            continue
        check(
            attr,
            set(x) == set(y) and all(np.array_equal(np.sort(x[k]), np.sort(y[k])) for k in y),
            f"{len(x):,} units",
        )
    check("pairs", a.pairs.equals(b.pairs), f"{len(a.pairs):,} rows")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path)
    parser.add_argument("--compare", type=Path, default=None)
    args = parser.parse_args()

    # Loaded twice on purpose: `all_cells=True` is the file as written, and the
    # default is what every analysis sees. On a plain export they are the same
    # object and the second pass costs nothing.
    full = load_export(args.export, all_cells=True)
    check_one(full, f"{args.export.name}, as written")

    if not np.asarray(full.in_use, dtype=bool).all():
        sub = subset_to_cells_in_use(full)
        check_one(sub, f"{args.export.name}, read normally")
        check(
            "subsetting lands on the QC-passing cell count",
            sub.metadata["n_cells"] == int(np.asarray(full.in_use, dtype=bool).sum()),
            f"{sub.metadata['n_cells']:,}",
        )
        # What the extra cells are there for: they move the per-gene mean the
        # simulation draws from.
        m_full = np.asarray(full.response_matrix.mean(axis=1)).ravel()
        m_sub = np.asarray(sub.response_matrix.mean(axis=1)).ravel()
        rel = np.abs(m_full - m_sub) / m_sub
        print(
            f"   raw gene mean shifts by {np.median(rel):.2%} at the median, "
            f"{rel.max():.2%} at most, when the QC-failed cells are included"
        )

    if args.compare is not None:
        other = load_export(args.compare)
        compare(load_export(args.export), other, args.export.name, args.compare.name)

    print("\n" + ("ALL PASS" if not FAILURES else f"{len(FAILURES)} FAILED: {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
