#!/usr/bin/env Rscript
# Extract a post-QC sceptre object out of R.
#
# Writes a columnar intermediate; run scripts/make_h5ad.py on the output
# directory to produce the dataset.h5ad pysceptre reads. h5ad is not written
# directly from R because rhdf5 writes length-1 attributes as arrays where
# anndata requires scalars, and R is the wrong side of the boundary to chase
# the on-disk spec from.
#
# Generalises the moi5-specific export_moi5_for_pysceptre.R: it takes any
# sceptre object plus its backing ondisc matrices, and is driven by flags
# rather than hardcoded paths.
#
# Two deliberate differences from the older script:
#
#   * The response matrix is written as a SPARSE TRIPLET, never densified. The
#     old script did `as.matrix()` on the whole thing, which needed ~38 GiB for
#     the full moi5 gene set and was OOM-killed. Genes are read one at a time
#     straight out of the odm (~0.4 ms each) and only their nonzero entries are
#     kept.
#   * Output is columnar with a metadata.json sidecar, rather than raw float64
#     .bin blobs plus .txt label files that carry no dtype, shape or column
#     information.
#
# Usage:
#   export_sceptre_dataset.R --sceptre-object so.rds --response-odm gene.odm \
#       --grna-odm grna.odm --out-dir out/ [--all-genes] [--discovery-result r.rds]
#
# Convert with scripts/make_h5ad.py, then read with sceptre_io.load_export.

suppressPackageStartupMessages({
  library(sceptre)
  library(ondisc)
})

source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])),
                 "sceptre_export_lib.R"))

parse_args <- function(argv) {
  opts <- list(all_genes = FALSE, discovery_result = NA_character_)
  i <- 1
  while (i <= length(argv)) {
    key <- argv[[i]]
    if (key == "--all-genes") {
      opts$all_genes <- TRUE
      i <- i + 1
      next
    }
    if (i + 1 > length(argv)) stop(sprintf("%s needs a value", key), call. = FALSE)
    value <- argv[[i + 1]]
    switch(key,
      "--sceptre-object"   = opts$sceptre_object <- value,
      "--response-odm"     = opts$response_odm <- value,
      "--grna-odm"         = opts$grna_odm <- value,
      "--out-dir"          = opts$out_dir <- value,
      "--discovery-result" = opts$discovery_result <- value,
      stop(sprintf("unknown argument: %s", key), call. = FALSE)
    )
    i <- i + 2
  }
  # --response-odm is only needed for ondisc-backed objects. An object whose
  # matrices are already in memory (a dgRMatrix in the response slot) is read
  # straight from the RDS.
  for (required in c("sceptre_object", "out_dir")) {
    if (is.null(opts[[required]])) {
      stop(sprintf("--%s is required", gsub("_", "-", required)), call. = FALSE)
    }
  }
  opts
}

main <- function() {
  opts <- parse_args(commandArgs(trailingOnly = TRUE))
  t_start <- Sys.time()

  if (is.null(opts$response_odm)) {
    cat("Reading in-memory sceptre object...\n")
    so <- readRDS(opts$sceptre_object)
    if (!length(sceptre:::get_response_matrix(so))) {
      stop("object has no response matrix; pass --response-odm for an ondisc-backed object",
           call. = FALSE)
    }
  } else {
    cat("Reattaching odm-backed matrices...\n")
    so <- sceptre:::read_ondisc_backed_sceptre_object(
      sceptre_object_fp = opts$sceptre_object,
      response_odm_file_fp = opts$response_odm,
      grna_odm_file_fp = if (is.null(opts$grna_odm)) opts$response_odm else opts$grna_odm
    )
  }
  if (!is.na(opts$discovery_result)) {
    loaded <- readRDS(opts$discovery_result)
    so@discovery_result <- if (is.data.frame(loaded)) loaded else loaded$discovery_result
  }
  export_sceptre_object(so, opts$out_dir, opts$all_genes,
                        source_label = normalizePath(opts$sceptre_object))
  cat("\nDone in", format(Sys.time() - t_start), "->", opts$out_dir, "\n")
}

main()
