#!/usr/bin/env Rscript
# Assemble everything scripts/score_power_against_simulation.py needs, from R.
#
# It comes from R rather than from an export because two of the three pieces live in sceptre slots
# a pysceptre export does not carry at this resolution:
#
#   per-gRNA cell counts   @initial_grna_assignment_list, restricted to @cells_in_use. That
#                          restriction is the whole subtlety: the slot is indexed against ALL
#                          cells (586,309 on day0) while everything an analysis uses is indexed
#                          against cells_in_use (567,690), so counting it raw would compare a
#                          treated count that includes QC-dropped cells against a threshold
#                          derived from those that remain. Same trap as measure_union_vs_sum.R.
#   baseline statistics    mean(exp(Z b)) and theta, taken straight off
#                          @response_precomputations so the estimate sits on the scale sceptre's
#                          own test operates on, with no refit and no chance of a different
#                          design matrix.
#
# The size-factor-normalised mean is dumped alongside, from a WattEG prepared/ directory, because
# the comparison is worth running under both expression conventions: they differ by about 16% on
# day0 and the whole question is whether that moves a decision taken at a 0.8 bar.
#
# Usage:
#   Rscript scripts/dump_day0_power_inputs.R <sceptre_object.rds> <prepared_dir> <out.json>

suppressPackageStartupMessages({library(sceptre); library(jsonlite)})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("need: <sceptre_object.rds> <prepared_dir> <out.json>", call. = FALSE)
object_path <- args[1]; prepared <- args[2]; out_path <- args[3]

so <- readRDS(object_path)
cells_in_use <- so@cells_in_use
n_all <- nrow(so@covariate_data_frame)
cat(sprintf("cells: %d total, %d in use\n", n_all, length(cells_in_use)))

# All-cells index -> cells_in_use index; NA for a cell QC removed.
to_in_use <- rep(NA_integer_, n_all)
to_in_use[cells_in_use] <- seq_along(cells_in_use)

initial <- so@initial_grna_assignment_list
kept_per_grna <- vapply(initial, function(idx) sum(!is.na(to_in_use[idx])), integer(1))

gtdf <- so@grna_target_data_frame[, c("grna_id", "grna_target")]
gtdf <- gtdf[gtdf$grna_target != "non-targeting", ]
gtdf$num_cells <- kept_per_grna[match(gtdf$grna_id, names(kept_per_grna))]
gtdf$num_cells[is.na(gtdf$num_cells)] <- 0L
cat(sprintf("design: %d (grna_id, grna_target) rows over %d targets\n",
            nrow(gtdf), length(unique(gtdf$grna_target))))

# Baseline on sceptre's own scale: the average fitted value, and the same fit's theta.
rp <- so@response_precomputations
Z <- so@covariate_matrix
Zu <- if (nrow(Z) == length(cells_in_use)) Z else Z[cells_in_use, , drop = FALSE]
genes <- names(rp)
model_mean <- vapply(genes, function(g) mean(exp(as.numeric(Zu %*% rp[[g]]$fitted_coefs))),
                     numeric(1))
theta <- vapply(genes, function(g) rp[[g]]$theta, numeric(1))
cat(sprintf("baseline: %d genes with a precomputation\n", length(genes)))

# The other convention, for the side-by-side.
si <- readRDS(file.path(prepared, "sim_input.rds"))
pos_genes <- rownames(si$row_data)
pos_mean <- si$row_data$mean
cat(sprintf("poscounts mean available for %d genes\n", length(pos_genes)))

threshold <- as.numeric(readLines(file.path(prepared, "discovery_threshold.txt"))[1])
cat(sprintf("nominal threshold: %.8g\n", threshold))

writeLines(jsonlite::toJSON(list(
  n_cells = length(cells_in_use),
  threshold = threshold,
  design = gtdf,
  model_genes = genes, model_mean = as.numeric(model_mean), theta = as.numeric(theta),
  poscounts_genes = pos_genes, poscounts_mean = as.numeric(pos_mean)
), digits = 17, auto_unbox = TRUE), out_path)
cat(sprintf("wrote %s\n", out_path))
