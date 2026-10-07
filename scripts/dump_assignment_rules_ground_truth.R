#!/usr/bin/env Rscript
# Ground truth for the ports of sceptre's thresholding and maximum assignment
# (pysceptre.assignment.thresholding, .maximum, .api and cells.cells_w_zero_or_twoplus_grnas).
#
#   Rscript scripts/dump_assignment_rules_ground_truth.R <out.json>
#
# Writes one JSON fixture; tests/validation/conftest.py gzips it to
# assignment_rules_ground_truth.json.gz (mtime 0). If you change this script, DELETE the .gz and
# regenerate it; the fixture records this script's md5 and a test compares it with the script
# on disk.
#
# Data: synthetic only, drawn with set.seed(20261007). 12 gRNAs (8 targeting, two per target,
# and 4 non-targeting) x 400 cells: each cell carries one gRNA at Poisson(8) over a Poisson(0.3)
# background, and the first cells are set by hand to sit on every cut the two methods have:
# empty cells, ties for the top gRNA (the first row wins), a top share of exactly 0.8 and either
# side of it, cells with exactly 4 and 5 gRNA UMIs, counts equal to each threshold, and a top
# gRNA in the last row. Every comparison is exact arithmetic on integers, so no value needs to be
# kept away from a cut.
#
# Runs: sceptre's public path, import_data -> set_analysis_parameters -> assign_grnas, for each
# method and hyperparameter set below, recording initial_grna_assignment_list and
# cells_w_zero_or_twoplus_grnas, plus the per-cell top gRNA and its share that import_data
# computes (import_grna_assignment_info) and the cells' gRNA UMIs; then the inputs sceptre
# refuses, each recorded as refused or not.
#
# Conventions: indices 0-based; the gRNA matrix as (grna, cell, count) triplets over its nonzero
# entries; non-finite numbers as "NaN", "Inf", "-Inf", "NA".

suppressPackageStartupMessages({
  library(Matrix)
  library(sceptre)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("usage: dump_assignment_rules_ground_truth.R <out.json>")
out_path <- args[[1]]
script_path <- normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)))

`%||%` <- function(a, b) if (is.null(a)) b else a
arr <- function(x) I(x)
num <- function(x) {
  x <- as.numeric(x)
  out <- as.list(x)
  out[is.na(x) & !is.nan(x)] <- "NA"
  out[is.nan(x)] <- "NaN"
  out[is.infinite(x) & x > 0] <- "Inf"
  out[is.infinite(x) & x < 0] <- "-Inf"
  out
}
pkg <- function(p) {
  d <- utils::packageDescription(p)
  list(version = d$Version, remote_sha = d$RemoteSha %||% NA_character_)
}

SEED <- 20261007L
N_CELLS <- 400L
N_GENES <- 10L
grna_ids <- c(sprintf("t%d_g%d", rep(1:4, each = 2), rep(1:2, 4)), sprintf("nt_%d", 1:4))
grna_targets <- c(rep(sprintf("target_%d", 1:4), each = 2), rep("non-targeting", 4))
N_GRNAS <- length(grna_ids)

set.seed(SEED)
g <- matrix(rpois(N_GRNAS * N_CELLS, 0.3), nrow = N_GRNAS)
carrier <- sample.int(N_GRNAS, N_CELLS, replace = TRUE)
g[cbind(carrier, seq_len(N_CELLS))] <- g[cbind(carrier, seq_len(N_CELLS))] + rpois(N_CELLS, 8)
response <- matrix(rpois(N_GENES * N_CELLS, 3), nrow = N_GENES)

# Cells set by hand (columns 1 to 20; each line is one cell's counts by gRNA row).
hand <- list(
  rep(0, 12),                                   # empty
  rep(0, 12),                                   # empty
  c(3, 3, rep(0, 10)),                          # tie between rows 1 and 2
  c(0, 0, 0, 6, 0, 0, 0, 0, 0, 6, 0, 0),        # tie between rows 4 and 10
  c(8, 2, rep(0, 10)),                          # top share exactly 0.8
  c(9, 2, rep(0, 10)),                          # top share above 0.8
  c(7, 2, rep(0, 10)),                          # top share below 0.8
  c(4, rep(0, 11)),                             # 4 gRNA UMIs
  c(5, rep(0, 11)),                             # 5 gRNA UMIs
  c(0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),        # a count equal to threshold 5
  c(0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0),        # a count equal to threshold 7
  c(1, rep(0, 11)),                             # a count equal to threshold 1
  c(rep(0, 11), 12),                            # top gRNA in the last row
  c(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),        # twelve-way tie
  c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2),        # a single small count, last row
  c(10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),      # tie at a count above every threshold
  c(16, 4, rep(0, 10)),                         # top share exactly 0.8 at 20 UMIs
  c(0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 1),        # top share exactly 0.9
  c(6, 6, 6, rep(0, 9)),                        # three-way tie
  c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)         # empty
)
for (k in seq_along(hand)) g[, k] <- hand[[k]]
storage.mode(g) <- "integer"
rownames(g) <- grna_ids
colnames(g) <- sprintf("cell_%d", seq_len(N_CELLS))
rownames(response) <- sprintf("gene_%d", seq_len(N_GENES))
colnames(response) <- colnames(g)
grna_matrix <- as(Matrix(g, sparse = TRUE), "CsparseMatrix")
response_matrix <- as(Matrix(response, sparse = TRUE), "CsparseMatrix")
grna_target_data_frame <- data.frame(grna_id = grna_ids, grna_target = grna_targets)

import <- function(moi) {
  so <- suppressMessages(import_data(
    response_matrix = response_matrix,
    grna_matrix = grna_matrix,
    grna_target_data_frame = grna_target_data_frame,
    moi = moi
  ))
  suppressMessages(set_analysis_parameters(so))
}
so_low <- import("low")
so_high <- import("high")

cells0 <- function(idx) arr(as.integer(sort(idx) - 1L))
record_run <- function(label, so, method, hyper) {
  res <- withCallingHandlers(
    suppressMessages(do.call(assign_grnas, c(list(sceptre_object = so, method = method, parallel = FALSE, print_progress = FALSE), hyper))),
    warning = function(w) invokeRestart("muffleWarning")
  )
  lst <- res@initial_grna_assignment_list
  stopifnot(identical(names(lst), grna_ids))
  list(
    label = label,
    moi = if (so@low_moi) "low" else "high",
    method = method,
    resolved_method = res@grna_assignment_method,
    hyperparameters = hyper,
    assignments = lapply(lst, cells0),
    cells_w_zero_or_twoplus_grnas = cells0(res@cells_w_zero_or_twoplus_grnas)
  )
}

runs <- list(
  record_run("maximum_default", so_low, "maximum", list()),
  record_run("maximum_frac0.5_umis0", so_low, "maximum", list(umi_fraction_threshold = 0.5, min_grna_n_umis_threshold = 0L)),
  record_run("maximum_frac0.9_umis10", so_low, "maximum", list(umi_fraction_threshold = 0.9, min_grna_n_umis_threshold = 10L)),
  record_run("default_low", so_low, "default", list()),
  record_run("thresholding_5_low", so_low, "thresholding", list(threshold = 5)),
  record_run("thresholding_1_low", so_low, "thresholding", list(threshold = 1)),
  record_run("thresholding_7_low", so_low, "thresholding", list(threshold = 7)),
  record_run("thresholding_5_high", so_high, "thresholding", list(threshold = 5)),
  record_run("thresholding_none_low", so_low, "thresholding", list(threshold = 1e6))
)

refused <- function(so, method, hyper) {
  out <- tryCatch({
    suppressMessages(do.call(assign_grnas, c(list(sceptre_object = so, method = method, parallel = FALSE, print_progress = FALSE), hyper)))
    FALSE
  }, error = function(e) TRUE)
  out
}
refusals <- list(
  list(label = "maximum_high", moi = "high", method = "maximum", hyperparameters = list(), refused = refused(so_high, "maximum", list())),
  list(label = "threshold_0.5", moi = "low", method = "thresholding", hyperparameters = list(threshold = 0.5), refused = refused(so_low, "thresholding", list(threshold = 0.5))),
  list(label = "frac_1", moi = "low", method = "maximum", hyperparameters = list(umi_fraction_threshold = 1, min_grna_n_umis_threshold = 5L), refused = refused(so_low, "maximum", list(umi_fraction_threshold = 1, min_grna_n_umis_threshold = 5L))),
  list(label = "frac_0", moi = "low", method = "maximum", hyperparameters = list(umi_fraction_threshold = 0, min_grna_n_umis_threshold = 5L), refused = refused(so_low, "maximum", list(umi_fraction_threshold = 0, min_grna_n_umis_threshold = 5L))),
  list(label = "umis_negative", moi = "low", method = "maximum", hyperparameters = list(umi_fraction_threshold = 0.8, min_grna_n_umis_threshold = -1L), refused = refused(so_low, "maximum", list(umi_fraction_threshold = 0.8, min_grna_n_umis_threshold = -1L)))
)

info <- so_low@import_grna_assignment_info
trip <- summary(grna_matrix)
out <- list(
  provenance = list(
    sceptre = pkg("sceptre"), Matrix = pkg("Matrix"), jsonlite = pkg("jsonlite"),
    r_version = R.version.string, platform = R.version$platform,
    dumper_md5 = unname(tools::md5sum(script_path)),
    seed = SEED
  ),
  n_grnas = N_GRNAS,
  n_cells = N_CELLS,
  grna_ids = arr(grna_ids),
  grna_targets = arr(grna_targets),
  counts = list(grna = arr(as.integer(trip$i - 1L)), cell = arr(as.integer(trip$j - 1L)), count = arr(as.numeric(trip$x))),
  max_grna = arr(match(info$max_grna, grna_ids) - 1L),
  max_grna_frac_umis = num(info$max_grna_frac_umis),
  grna_n_umis = num(so_low@covariate_data_frame$grna_n_umis),
  runs = runs,
  refusals = refusals
)

json <- jsonlite::toJSON(out, auto_unbox = TRUE, digits = I(17), na = "string", null = "null")
writeLines(json, out_path)
cat("wrote", out_path, format(file.size(out_path), big.mark = ","), "bytes\n")
