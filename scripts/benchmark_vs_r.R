#!/usr/bin/env Rscript
# Per-step benchmark of R sceptre, and the source of truth for pysceptre's
# inputs.
#
# Each step runs as its OWN process so /usr/bin/time -l reports a true peak RSS
# per step rather than a high-water mark for the whole pipeline. Drive it with
# scripts/benchmark_vs_r.sh.
#
#   prepare     materialize out of ondisc, set CRT parameters, assign gRNAs,
#               run QC, save the post-QC object, and EXPORT the dataset
#   discovery   run_discovery_analysis
#   calibration run_calibration_check
#   power       run_power_check
#
# `prepare` exports the exact object the other steps analyze. Everything
# downstream -- R's own results and pysceptre's inputs -- derives from that one
# object, so the two implementations cannot silently diverge on their inputs.
# They did before this existed: re-deriving gRNA assignments reproduced only
# 39 of 2,974 targets' cell sets.
#
# Usage: benchmark_vs_r.R <step> <data_dir> <out_dir>

suppressPackageStartupMessages({
  library(sceptre)
  library(ondisc)
  library(Matrix)
  library(jsonlite)
})

script_dir <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
source(file.path(script_dir, "sceptre_export_lib.R"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("usage: benchmark_vs_r.R <step> <data_dir> <out_dir>", call. = FALSE)
step <- args[[1]]; data_dir <- args[[2]]; out_dir <- args[[3]]
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

say <- function(...) { cat(format(Sys.time(), "%H:%M:%S"), "|", ..., "\n"); flush.console() }
postqc_fp <- file.path(out_dir, "so_postqc_crt.rds")

#' Record wall and CPU time for one step as JSON. Peak RSS is captured
#' externally by /usr/bin/time -l, because R cannot observe its own RSS
#' portably (an earlier attempt using gc() reported 529,301 GB).
timed <- function(label, expr) {
  t0 <- Sys.time()
  cpu <- system.time(value <- force(expr))
  wall <- as.numeric(Sys.time() - t0, units = "secs")
  timing <- list(step = label, wall_seconds = wall,
                 user_seconds = unname(cpu[["user.self"]]),
                 sys_seconds = unname(cpu[["sys.self"]]),
                 r_version = as.character(getRversion()),
                 sceptre_version = as.character(packageVersion("sceptre")),
                 finished_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"))
  write_json(timing, file.path(out_dir, sprintf("timing_r_%s.json", label)),
             auto_unbox = TRUE, pretty = TRUE)
  say(sprintf("  %s: wall %.1fs (%.2f min), user %.1fs, sys %.1fs",
              label, wall, wall / 60, timing$user_seconds, timing$sys_seconds))
  value
}

materialize_odm <- function(odm, label) {
  # sceptre refuses CRT on an ondisc-backed object, so the matrices must come
  # into memory first. ondisc allows single-index access only, so rows are read
  # one at a time and accumulated as triplets -- never densified. Built as
  # dgRMatrix (row-compressed) because sceptre reads @j/@p/@x.
  n_rows <- nrow(odm)
  js <- vector("list", n_rows); xs <- vector("list", n_rows); counts <- integer(n_rows)
  for (k in seq_len(n_rows)) {
    v <- odm[k, ]; nz <- which(v != 0)
    js[[k]] <- nz; xs[[k]] <- as.double(v[nz]); counts[[k]] <- length(nz)
  }
  say("  ", label, "materialized:", format(sum(counts), big.mark = ","), "nonzeros")
  methods::new("dgRMatrix",
    j = as.integer(unlist(js, use.names = FALSE)) - 1L,
    p = c(0L, cumsum(counts)),
    x = unlist(xs, use.names = FALSE),
    Dim = as.integer(dim(odm)), Dimnames = dimnames(odm))
}

if (step == "prepare") {
  say("reattaching odm-backed object")
  so <- sceptre:::read_ondisc_backed_sceptre_object(
    sceptre_object_fp = file.path(data_dir, "sceptre_object.rds"),
    response_odm_file_fp = file.path(data_dir, "gene.odm"),
    grna_odm_file_fp = file.path(data_dir, "grna.odm"))

  timed("prepare", {
    so <- sceptre:::set_response_matrix(so, materialize_odm(sceptre:::get_response_matrix(so), "response"))
    so <- sceptre:::set_grna_matrix(so, materialize_odm(sceptre:::get_grna_matrix(so), "grna"))

    # The input discovery_pairs slot is empty (the Nextflow pipeline supplied
    # pairs separately); discovery_pairs_with_info retains every pair.
    dpi <- so@discovery_pairs_with_info
    pairs_in <- data.frame(grna_target = as.character(dpi$grna_group),
                           response_id = as.character(dpi$response_id),
                           stringsAsFactors = FALSE)
    say("reconstructed", nrow(pairs_in), "input discovery pairs")

    # Positive control pairs are likewise absent from the saved object, but
    # ship alongside it; without them run_power_check has nothing to test.
    pc_fp <- file.path(data_dir, "positive.rds")
    pc_pairs <- if (file.exists(pc_fp)) {
      pc <- readRDS(pc_fp)
      say("loaded", nrow(pc), "positive control pairs")
      pc[, c("grna_target", "response_id")]
    } else {
      say("NOTE: no positive.rds found -- power check will be unavailable")
      data.frame(grna_target = character(), response_id = character())
    }

    so <- sceptre::set_analysis_parameters(so,
      discovery_pairs = pairs_in, positive_control_pairs = pc_pairs, side = "left",
      grna_integration_strategy = "union", formula_object = so@formula_object,
      resampling_approximation = "skew_normal", resampling_mechanism = "crt",
      multiple_testing_alpha = 0.1)
    so <- sceptre::assign_grnas(so, method = "thresholding", threshold = 5, parallel = FALSE)
    so <- sceptre::run_qc(so)
    so
  })
  say("B1/B2/B3 =", so@B1, "/", so@B2, "/", so@B3,
      "| permutations =", so@run_permutations,
      "| ok pairs =", so@n_ok_discovery_pairs,
      "| memberships =", sum(lengths(so@grna_assignments$grna_group_idxs)))

  saveRDS(so, postqc_fp)
  say("exporting the exact analyzed object")
  export_sceptre_object(so, file.path(out_dir, "export"),
                        source_label = normalizePath(postqc_fp))
  say("prepare done ->", out_dir)

} else {
  if (!file.exists(postqc_fp)) stop("run `prepare` first", call. = FALSE)
  say("loading post-QC object")
  so <- readRDS(postqc_fp)

  if (step == "discovery") {
    so <- timed("discovery", sceptre::run_discovery_analysis(so, parallel = FALSE))
    result <- so@discovery_result
  } else if (step == "calibration") {
    so <- timed("calibration", sceptre::run_calibration_check(so, parallel = FALSE))
    result <- so@calibration_result
  } else if (step == "power") {
    so <- timed("power", sceptre::run_power_check(so, parallel = FALSE))
    result <- so@power_result
  } else {
    stop(sprintf("unknown step: %s", step), call. = FALSE)
  }

  arrow::write_parquet(as.data.frame(result), file.path(out_dir, sprintf("r_%s_result.parquet", step)))
  say(step, "result:", nrow(result), "rows ->", out_dir)
}
