"""Turn the R extraction into pysceptre's input format, an .h5ad.

pysceptre reads h5ad. That is what the single-cell Python ecosystem uses
(`anndata`, `scanpy`, `muon`), and it stores sparse matrices in CSR/CSC form
natively, so a load reads straight into the final arrays -- measured on moi5,
0.41 GB against 1.17 GB for the intermediate columnar form, and a smaller file
(62 MB against 76 MB).

`export_sceptre_dataset.R` cannot write h5ad directly: rhdf5 writes length-1
attributes as arrays where anndata requires scalars, and chasing the exact
on-disk spec from R is effort spent on the wrong side of the boundary. R is
only involved to get a dataset out of `ondisc` once; pysceptre users bring
their own AnnData. So R writes a plain intermediate and this converts it,
removing the intermediate afterwards so the dataset is a single file.

Usage: make_h5ad.py <extraction_dir> [--keep-intermediate]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sceptre_io import _load_intermediate, write_h5ad  # noqa: E402

INTERMEDIATE_FILES = (
    "response_matrix.parquet",
    "gene_ids.parquet",
    "covariate_matrix.parquet",
    "grna_target_cells.parquet",
    "pairs.parquet",
    "discovery_result.parquet",
)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    directory = Path(sys.argv[1])
    keep = "--keep-intermediate" in sys.argv

    export = _load_intermediate(directory)
    out = write_h5ad(export, directory / "dataset.h5ad")
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"  {export.describe()}")

    if keep:
        print("  keeping the intermediate files (--keep-intermediate)")
        return
    removed = 0
    for name in INTERMEDIATE_FILES:
        path = directory / name
        if path.exists():
            path.unlink()
            removed += 1
    print(f"  removed {removed} intermediate files; dataset.h5ad is the dataset")


if __name__ == "__main__":
    main()
