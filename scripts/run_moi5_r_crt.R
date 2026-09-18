#!/usr/bin/env Rscript
# Run R sceptre's discovery analysis on the moi5 object with the CRT resampling
# mechanism, single-core, for a like-for-like comparison against pysceptre.
#
# The Seqera run of this dataset used permutations (the sceptre Nextflow
# pipeline does not offer CRT), drawing B1+B2+B3 = 499+4999+24999 = 30,497
# samples per target. pysceptre implements CRT only, which draws 5,498. This
# script closes that gap by re-running the same object under CRT.
#
# Single-core deliberately: sceptre forks one process per processor, each
# holding its own copy of the data, so n_processors=13 needs roughly 13x the
# memory. pysceptre is single-threaded (and pins BLAS to one thread), so
# single-core R is also the fair comparison.
#
# Usage: run_moi5_r_crt.R <data_dir> <out_dir>

suppressPackageStartupMessages({
  library(sceptre)
  library(ondisc)
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: run_moi5_r_crt.R <data_dir> <out_dir>", call. = FALSE)
data_dir <- args[[1]]
out_dir <- args[[2]]
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

say <- function(...) { cat(format(Sys.time(), "%H:%M:%S"), "|", ..., "\n"); flush.console() }

materialize_odm <- function(odm, label) {
  # sceptre refuses CRT on an ondisc-backed object:
  #   "`resampling_mechanism` must be set to 'permutations' when using an
  #    ondisc-backed sceptre_object"
  # which is why the Nextflow pipeline can only run permutations. Pulling the
  # matrix into memory as a sparse dgCMatrix lifts that restriction. ondisc
  # allows single-index access only, so rows are read one at a time (~0.8 ms
  # each) and accumulated as triplets -- never densified.
  n_rows <- nrow(odm)
  js <- vector("list", n_rows)
  xs <- vector("list", n_rows)
  counts <- integer(n_rows)
  t0 <- Sys.time()
  for (k in seq_len(n_rows)) {
    v <- odm[k, ]
    nz <- which(v != 0)
    js[[k]] <- nz
    xs[[k]] <- as.double(v[nz])
    counts[[k]] <- length(nz)
    if (k %% 10000 == 0) say("   ", label, k, "/", n_rows)
  }
  # Build dgRMatrix (row-compressed) directly, not dgCMatrix: sceptre's
  # compute_nt_nonzero_matrix_and_n_ok_pairs_v3 reads @j/@p/@x, which only a
  # row-compressed matrix has. Accumulating row-wise makes this the natural
  # layout anyway -- no conversion pass needed.
  mat <- methods::new(
    "dgRMatrix",
    j = as.integer(unlist(js, use.names = FALSE)) - 1L,  # 0-based columns
    p = c(0L, cumsum(counts)),                           # row pointers
    x = unlist(xs, use.names = FALSE),
    Dim = as.integer(dim(odm)),
    Dimnames = dimnames(odm)
  )
  say("  ", label, "materialized:", format(sum(counts), big.mark = ","),
      "nonzeros in", format(Sys.time() - t0))
  mat
}

say("reattaching odm-backed object")
so <- sceptre:::read_ondisc_backed_sceptre_object(
  sceptre_object_fp = file.path(data_dir, "sceptre_object.rds"),
  response_odm_file_fp = file.path(data_dir, "gene.odm"),
  grna_odm_file_fp = file.path(data_dir, "grna.odm")
)

say("materializing response matrix into memory (required for CRT)")
so <- sceptre:::set_response_matrix(
  so, materialize_odm(sceptre:::get_response_matrix(so), "response")
)
say("materializing grna matrix into memory")
so <- sceptre:::set_grna_matrix(
  so, materialize_odm(sceptre:::get_grna_matrix(so), "grna")
)

# The input `discovery_pairs` slot is empty (the Nextflow pipeline supplied
# pairs separately), but discovery_pairs_with_info retains every pair that was
# considered, so the original input is recoverable exactly.
dpi <- so@discovery_pairs_with_info
discovery_pairs <- data.frame(
  grna_target = as.character(dpi$grna_group),
  response_id = as.character(dpi$response_id),
  stringsAsFactors = FALSE
)
say("reconstructed", nrow(discovery_pairs), "input discovery pairs")

# Changing analysis parameters invalidates assign_grnas/run_qc, so both are
# re-run. The object records thresholding with threshold=5; reproduce it.
say("set_analysis_parameters(resampling_mechanism = 'crt')")
so <- sceptre::set_analysis_parameters(
  so,
  discovery_pairs = discovery_pairs,
  side = "left",
  grna_integration_strategy = "union",
  formula_object = so@formula_object,
  resampling_approximation = "skew_normal",
  resampling_mechanism = "crt",
  multiple_testing_alpha = 0.1
)

say("assign_grnas(thresholding, threshold = 5)")
t0 <- Sys.time()
so <- sceptre::assign_grnas(so, method = "thresholding", threshold = 5, parallel = FALSE)
say("  assign_grnas took", format(Sys.time() - t0))

say("run_qc()")
t0 <- Sys.time()
so <- sceptre::run_qc(so)
say("  run_qc took", format(Sys.time() - t0))
say("  B1/B2/B3 =", so@B1, "/", so@B2, "/", so@B3,
    "| run_permutations =", so@run_permutations,
    "| ok discovery pairs =", so@n_ok_discovery_pairs)

say("run_discovery_analysis() -- single core, this is the timed comparison")
t0 <- Sys.time()
so <- sceptre::run_discovery_analysis(so, parallel = FALSE, print_progress = TRUE)
elapsed <- Sys.time() - t0
say("  run_discovery_analysis took", format(elapsed))

result <- so@discovery_result
saveRDS(result, file.path(out_dir, "r_crt_discovery_result.rds"))
write.csv(result, file.path(out_dir, "r_crt_discovery_result.csv"), row.names = FALSE)

secs <- as.numeric(elapsed, units = "secs")
say(sprintf("WALL %.1fs = %.2f min for %d pairs (%.2f ms/pair)",
            secs, secs / 60, nrow(result), secs / nrow(result) * 1000))
say("significant:", sum(result$significant, na.rm = TRUE),
    "| p < 1e-4:", sum(result$p_value < 1e-4, na.rm = TRUE))
say("peak RSS (GB):", round(as.numeric(gc()[2, 6]) / 1024, 2))
say("wrote", file.path(out_dir, "r_crt_discovery_result.csv"))
