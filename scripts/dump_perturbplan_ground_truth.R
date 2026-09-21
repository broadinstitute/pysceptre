#!/usr/bin/env Rscript
# Ground truth for the analytical power port, from PerturbPlan's own R.
#
# Writes a JSON fixture that `tests/validation/test_analytical_power.py` compares against.
# `tests/validation/conftest.py` regenerates it only if the file is missing, so the suite runs in
# CI with no R and no perturbplan installed -- same contract as ground_truth.json. If you change
# the cases below, DELETE the JSON or the tests keep validating against the stale fixture.
#
# The fixture records perturbplan's version and commit SHA. ground_truth.json records neither for
# sceptre and CLAUDE.md notes the regret; this does not repeat it.
#
# On the SHA that matters: the comparison this port was validated against installed perturbplan
# from an unpinned clone, so its version was not recorded. It does not matter here. R/power_posthoc.R,
# R/posthoc_help.R and R/QC_computation.R -- which hold every function this port touches -- are
# byte-identical between 43232419fe26 (2026-02-08) and upstream HEAD 615ef7bb05a7 (2026-04-13,
# still the latest as of 2026-09-21). If a future regeneration reports a SHA outside that range,
# diff those three files before trusting the comparison.
#
# Usage:
#   Rscript scripts/dump_perturbplan_ground_truth.R tests/validation/perturbplan_ground_truth.json
#
# Install perturbplan first:
#   Rscript -e 'remotes::install_github("Katsevich-Lab/perturbplan")'

suppressMessages(library(perturbplan))

args <- commandArgs(trailingOnly = TRUE)
out_path <- if (length(args) >= 1) args[1] else "tests/validation/perturbplan_ground_truth.json"

pp_version <- as.character(utils::packageVersion("perturbplan"))
pp_sha <- tryCatch({
  d <- utils::packageDescription("perturbplan")
  sha <- d$GithubSHA1
  if (is.null(sha)) sha <- d$RemoteSha
  if (is.null(sha)) "unknown" else sha
}, error = function(e) "unknown")
cat(sprintf("perturbplan %s (%s)\n", pp_version, pp_sha))

# ---------------------------------------------------------------------------------------------
# The cases. Chosen to span the corners where a port is most likely to diverge:
#
#   one_grna        a single-gRNA target -- num_trt_cells_sq == num_trt_cells^2, so the
#                   across-gRNA variance term is at its maximum relative to the within term
#   many_grnas      20 gRNAs of unequal size, the real high-MOI shape, where num_trt_cells_sq
#                   is far below num_trt_cells^2
#   low_expression  a gene near the zero-inflation boundary, which is where compute_zero_prob
#                   and therefore the QC factor does real work
#   high_expression a gene where mean^2/size dominates var_nb
#   large_trt       a target perturbing a large share of all cells, so 1/n_cntrl and 1/n_trt are
#                   comparable rather than the usual n_trt << n_cntrl
#   tiny_trt        three cells, where the binomial QC factor is far from 1
#
# Every case is crossed with both `side` values, their matching cutoffs, and both QC-threshold
# settings (0, the setting WattEG scored, and 7, perturbplan's default), so the fixture pins the
# QC factor independently of the normal tail.
# ---------------------------------------------------------------------------------------------

n_total_cells <- 20000L

grnas <- list(
  one_grna = data.frame(
    grna_id = "g_one_1",
    grna_target = "one_grna",
    num_cells = 250L
  ),
  many_grnas = data.frame(
    grna_id = sprintf("g_many_%02d", 1:20),
    grna_target = "many_grnas",
    # Unequal on purpose: equal sizes make num_trt_cells_sq a function of the count alone and
    # would not catch a port that summed the wrong thing.
    num_cells = c(12L, 18L, 7L, 31L, 25L, 9L, 44L, 15L, 22L, 5L,
                  38L, 11L, 27L, 19L, 8L, 33L, 14L, 21L, 6L, 29L)
  ),
  large_trt = data.frame(
    grna_id = sprintf("g_large_%d", 1:4),
    grna_target = "large_trt",
    num_cells = c(2000L, 2500L, 1800L, 2200L)
  ),
  tiny_trt = data.frame(
    grna_id = "g_tiny_1",
    grna_target = "tiny_trt",
    num_cells = 3L
  )
)
cells_per_grna <- do.call(rbind, grnas)
rownames(cells_per_grna) <- NULL

# expression_size is the NB size, i.e. 1/dispersion -- theta, not the dispersion.
baseline <- data.frame(
  response_id = c("gene_low", "gene_mid", "gene_high"),
  expression_mean = c(0.03, 1.4, 85.0),
  expression_size = c(0.25, 1.1, 6.0),
  stringsAsFactors = FALSE
)

discovery_pairs <- expand.grid(
  grna_target = unique(cells_per_grna$grna_target),
  response_id = baseline$response_id,
  stringsAsFactors = FALSE
)[, c("grna_target", "response_id")]

# alpha here stands in for a screen's own nominal threshold. The two values are measured nominal
# thresholds from two real designs on one screen, an order of magnitude apart because one tests far
# more pairs than the other, so the fixture pins the far tail at both.
settings <- list(
  list(label = "left_cis", side = "left", cutoff = 6.4840e-4 / 2, trt_thresh = 0L, cntrl_thresh = 0L),
  list(label = "left_trans", side = "left", cutoff = 6.2252e-5 / 2, trt_thresh = 0L, cntrl_thresh = 0L),
  list(label = "left_qc7", side = "left", cutoff = 6.4840e-4 / 2, trt_thresh = 7L, cntrl_thresh = 7L),
  list(label = "both_cis", side = "both", cutoff = 6.4840e-4, trt_thresh = 0L, cntrl_thresh = 0L),
  list(label = "both_qc7", side = "both", cutoff = 6.4840e-4, trt_thresh = 7L, cntrl_thresh = 7L),
  list(label = "right_cis", side = "right", cutoff = 6.4840e-4, trt_thresh = 0L, cntrl_thresh = 0L)
)

# fold_change_mean is a MULTIPLIER: a 15 % knockdown is 0.85, not 0.15. fold_change_sd = 0.13 is
# the value WattEG scored, taken from perturbplan's own documentation example -- supplied, not
# derived, which is why the port requires it rather than defaulting to it.
effect_sizes <- c(0.05, 0.15, 0.50)
fold_change_sd <- 0.13

json_escape <- function(x) gsub('"', '\\\\"', x, fixed = FALSE)

num_fmt <- function(v) {
  # 17 significant digits round-trips a double exactly, so the fixture is not the thing that
  # loses precision when the comparison tolerance is ~1e-9.
  out <- vapply(v, function(z) {
    if (is.na(z)) "null" else formatC(z, digits = 17, format = "g")
  }, character(1))
  paste(out, collapse = ", ")
}

str_array <- function(v) paste0('"', json_escape(v), '"', collapse = ", ")

cases <- character(0)
for (st in settings) {
  for (es in effect_sizes) {
    res <- compute_power_posthoc(
      discovery_pairs = discovery_pairs,
      cells_per_grna = cells_per_grna,
      baseline_expression_stats = baseline,
      control_group = "complement",
      fold_change_mean = 1 - es,
      fold_change_sd = fold_change_sd,
      num_total_cells = n_total_cells,
      cutoff = st$cutoff,
      n_nonzero_trt_thresh = st$trt_thresh,
      n_nonzero_cntrl_thresh = st$cntrl_thresh,
      side = st$side
    )
    ip <- res$individual_power
    cases <- c(cases, sprintf(
      paste0('    {\n',
             '      "label": "%s_es%02d",\n',
             '      "side": "%s",\n',
             '      "cutoff": %s,\n',
             '      "effect_size": %s,\n',
             '      "fold_change_mean": %s,\n',
             '      "fold_change_sd": %s,\n',
             '      "n_nonzero_trt_thresh": %d,\n',
             '      "n_nonzero_cntrl_thresh": %d,\n',
             '      "grna_target": [%s],\n',
             '      "response_id": [%s],\n',
             '      "power": [%s],\n',
             '      "expected_num_discoveries": %s\n',
             '    }'),
      st$label, as.integer(round(es * 100)), st$side, num_fmt(st$cutoff), num_fmt(es),
      num_fmt(1 - es), num_fmt(fold_change_sd), st$trt_thresh, st$cntrl_thresh,
      str_array(ip$grna_target), str_array(ip$response_id), num_fmt(ip$power),
      num_fmt(res$expected_num_discoveries)
    ))
    cat(sprintf("  %-12s es=%.2f  power range [%.6g, %.6g]\n", st$label, es,
                min(ip$power), max(ip$power)))
  }
}

lines <- c(
  "{",
  sprintf('  "perturbplan_version": "%s",', pp_version),
  sprintf('  "perturbplan_sha": "%s",', pp_sha),
  sprintf('  "r_version": "%s",', paste(R.version$major, R.version$minor, sep = ".")),
  sprintf('  "control_group": "complement",'),
  sprintf('  "num_total_cells": %d,', n_total_cells),
  '  "cells_per_grna": {',
  sprintf('    "grna_id": [%s],', str_array(cells_per_grna$grna_id)),
  sprintf('    "grna_target": [%s],', str_array(cells_per_grna$grna_target)),
  sprintf('    "num_cells": [%s]', num_fmt(cells_per_grna$num_cells)),
  "  },",
  '  "baseline_expression_stats": {',
  sprintf('    "response_id": [%s],', str_array(baseline$response_id)),
  sprintf('    "expression_mean": [%s],', num_fmt(baseline$expression_mean)),
  sprintf('    "expression_size": [%s]', num_fmt(baseline$expression_size)),
  "  },",
  '  "discovery_pairs": {',
  sprintf('    "grna_target": [%s],', str_array(discovery_pairs$grna_target)),
  sprintf('    "response_id": [%s]', str_array(discovery_pairs$response_id)),
  "  },",
  '  "cases": [',
  paste(cases, collapse = ",\n"),
  "  ]",
  "}"
)
writeLines(lines, out_path)
cat(sprintf("wrote %s (%d cases)\n", out_path, length(cases)))
