#!/usr/bin/env Rscript
# Regenerate the fishash paper's simulated datasets and export them for Python.
#
# Usage:
#   Rscript scripts/fishash_eval/simulate.R <upstream_dir> <scenario> <sims_dir> [max_files]
#
# <upstream_dir> is the checkout made by fetch_upstream.sh. <scenario> names one of its
# simulate/simulate_<scenario>.R scripts, e.g. numCells20k_numGuides200_varyMOI. That script
# is sourced unmodified, with two seams in the environment it runs in:
#   - commandArgs() returns <sims_dir>/<scenario>/rds, its output directory;
#   - saveRDS() records each file it is asked to write, and stops the script before the
#     (max_files + 1)-th dataset is simulated. Its first argument is a promise, so returning
#     without forcing it means that dataset is never drawn.
# Each scenario draws all of its datasets in order from a single set.seed(), so the first
# max_files files are identical whether or not the run stops there. varyNumGuides puts its
# 80,000-guide datasets last; max_files = 40 keeps the 20 to 20,000-guide ones.
#
# Outputs, under <sims_dir>/<scenario>/:
#   rds/<sim_label>.Rds          the upstream script's own output, untouched
#   <sim_label>/counts.parquet, ground_truth.parquet, counts_signal.parquet
#                                (guide, cell[, value]) triplets, 0-based, column-major
#   <sim_label>/meta.json        provenance and summary numbers
#   manifest.json                every dataset in the order it was drawn
# A scenario whose manifest says "ok" is not simulated again; a dataset whose meta.json exists
# is not exported again.

suppressPackageStartupMessages({
  library(Matrix)
  library(SummarizedExperiment)
})

script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "eval_common.R"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3 || length(args) > 4) {
  stop("usage: simulate.R <upstream_dir> <scenario> <sims_dir> [max_files]")
}
upstream_dir <- normalizePath(args[[1]], mustWork = TRUE)
scenario <- args[[2]]
sims_dir <- args[[3]]
max_files <- if (length(args) == 4) as.integer(args[[4]]) else NA_integer_

upstream_script <- file.path(upstream_dir, "simulate", paste0("simulate_", scenario, ".R"))
if (!file.exists(upstream_script)) stop("no upstream script ", upstream_script)
upstream_sha <- system2("git", c("-C", shQuote(upstream_dir), "rev-parse", "HEAD"), stdout = TRUE)

scenario_dir <- file.path(sims_dir, scenario)
rds_dir <- file.path(scenario_dir, "rds")
refuse_unless_ignored(scenario_dir)
dir.create(rds_dir, recursive = TRUE, showWarnings = FALSE)
manifest_path <- file.path(scenario_dir, "manifest.json")

simulate_scenario <- function() {
  written <- character()
  stop_class <- "simulation_limit"
  env <- new.env(parent = globalenv())
  env$commandArgs <- function(trailingOnly = FALSE) rds_dir
  env$saveRDS <- function(object, file = "", ...) {
    if (!is.na(max_files) && length(written) >= max_files) {
      stop(structure(class = c(stop_class, "condition"), list(message = "limit reached", call = NULL)))
    }
    base::saveRDS(object, file = file, ...)
    written <<- c(written, file)
    stamp("wrote ", basename(file))
    invisible(NULL)
  }
  completed <- tryCatch(
    {
      sys.source(upstream_script, envir = env)
      TRUE
    },
    simulation_limit = function(e) FALSE
  )
  list(files = written, completed_upstream = completed)
}

manifest_ok <- function() {
  if (!file.exists(manifest_path)) return(FALSE)
  m <- jsonlite::read_json(manifest_path)
  identical(m$status, "ok") && identical(m$max_files, if (is.na(max_files)) "all" else max_files) &&
    all(file.exists(file.path(rds_dir, vapply(m$datasets, function(d) d$rds, character(1)))))
}

if (!manifest_ok()) {
  stamp("simulating ", scenario, if (!is.na(max_files)) paste0(" (first ", max_files, " datasets)") else "")
  sim <- simulate_scenario()
  datasets <- lapply(seq_along(sim$files), function(i) {
    list(index = i, sim_label = sub("\\.Rds$", "", basename(sim$files[[i]])), rds = basename(sim$files[[i]]))
  })
  write_json_atomic(list(
    scenario = scenario,
    status = "ok",
    max_files = if (is.na(max_files)) "all" else max_files,
    completed_upstream = sim$completed_upstream,
    upstream_sha = upstream_sha,
    upstream_script = basename(upstream_script),
    versions = versions(),
    datasets = datasets
  ), manifest_path)
} else {
  stamp("skip simulation of ", scenario, ": manifest is complete")
}

manifest <- jsonlite::read_json(manifest_path)
for (d in manifest$datasets) {
  out_dir <- file.path(scenario_dir, d$sim_label)
  if (file.exists(file.path(out_dir, "meta.json"))) next
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  rds_path <- file.path(rds_dir, d$rds)
  se <- readRDS(rds_path)
  counts <- assay(se, "counts")
  truth <- assay(se, "ground_truth")
  signal <- assay(se, "counts_signal")
  stopifnot(identical(dim(counts), dim(truth)), identical(dim(counts), dim(signal)))
  stopifnot(identical(rownames(se), paste0("feature_", seq_len(nrow(se)))))
  stopifnot(identical(colnames(se), paste0("cell_", seq_len(ncol(se)))))
  write_triplets(counts, file.path(out_dir, "counts.parquet"))
  write_triplets(truth, file.path(out_dir, "ground_truth.parquet"), logical = TRUE)
  write_triplets(signal, file.path(out_dir, "counts_signal.parquet"))
  truth_c <- as(truth, "CsparseMatrix")
  # counts > 0 stays sparse; counts == 0 would be a dense n_guides x n_cells matrix.
  n_true_nonzero <- sum(truth_c & (counts > 0))
  col_totals <- Matrix::colSums(counts)
  write_json_atomic(list(
    scenario = scenario,
    sim_label = d$sim_label,
    index = d$index,
    n_guides = nrow(counts),
    n_cells = ncol(counts),
    guide_names = "feature_<1..n_guides>",
    cell_names = "cell_<1..n_cells>",
    nnz_counts = length(as(counts, "CsparseMatrix")@x),
    sum_counts = sum(counts),
    nnz_truth = length(truth_c@x),
    n_true_zero_count = length(truth_c@x) - n_true_nonzero,
    median_cell_total = stats::median(col_totals),
    rds = d$rds,
    rds_sha256 = sha256_file(rds_path),
    upstream_sha = manifest$upstream_sha,
    versions = versions()
  ), file.path(out_dir, "meta.json"))
  stamp("exported ", d$sim_label)
}
stamp("done ", scenario, ": ", length(manifest$datasets), " datasets")
