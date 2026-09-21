#!/usr/bin/env Rscript
# How far apart are a target's SUMMED per-gRNA cell count and its UNION?
#
# analytical_power/ ports PerturbPlan's compute_power_posthoc(), whose
# num_trt_cells is the sum over a target's gRNAs, while pysceptre's own data model carries the
# union (one index array per target). The two are not interchangeable and substituting one for the
# other silently produces a different, unvalidated estimator. This reports how large that
# difference actually is, so docs/design.md can state a number instead of an argument.
#
# THE TWO SLOTS ARE INDEXED AGAINST DIFFERENT CELL SETS, which is the whole reason this needs a
# script rather than a one-liner:
#
#   @initial_grna_assignment_list   one entry per individual gRNA, indexed against ALL cells
#   @grna_assignments$grna_group_idxs   one entry per target, indexed against @cells_in_use
#
# On day0 that is 586,309 against 567,690. Summing the first and dividing by the second compares a
# count that includes cells QC dropped against one that does not, which inflates the ratio for
# reasons that have nothing to do with multiplicity. The individual assignments are therefore
# restricted to cells_in_use first, exactly as scripts/sceptre_export_lib.R does when it writes
# them out.
#
# Takes the path from the command line with no default: this reads a real screen, which is never
# committed, and a wrong-but-plausible default is worse than a missing one.
#
# Usage:
#   Rscript scripts/measure_union_vs_sum.R <sceptre_object.rds> [label]

suppressMessages(library(sceptre))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("usage: measure_union_vs_sum.R <sceptre_object.rds> [label]", call. = FALSE)
}
path <- args[1]
label <- if (length(args) >= 2) args[2] else basename(dirname(path))
if (!file.exists(path)) stop("no such file: ", path, call. = FALSE)

so <- readRDS(path)
if (!methods::is(so, "sceptre_object")) {
  stop(path, " is a ", paste(class(so), collapse = "/"), ", not a sceptre_object.", call. = FALSE)
}

individual <- so@initial_grna_assignment_list
union_idxs <- so@grna_assignments$grna_group_idxs
if (is.null(individual) || length(individual) == 0) {
  stop("@initial_grna_assignment_list is empty; assign_grnas() has not been called.", call. = FALSE)
}
if (is.null(union_idxs) || length(union_idxs) == 0) {
  stop("@grna_assignments$grna_group_idxs is empty; run_qc() has not been called.", call. = FALSE)
}

cells_in_use <- so@cells_in_use
n_all <- nrow(so@covariate_data_frame)

# All-cells index -> cells_in_use index, NA for a cell QC dropped. The individual assignments are
# counted only over cells that survive, which is the only way the sum and the union are on the
# same footing.
to_in_use <- rep(NA_integer_, n_all)
to_in_use[cells_in_use] <- seq_along(cells_in_use)

grna_map <- so@grna_target_data_frame
needed <- c("grna_id", "grna_target")
if (!all(needed %in% colnames(grna_map))) {
  stop("@grna_target_data_frame lacks: ", paste(setdiff(needed, colnames(grna_map)), collapse = ", "),
       call. = FALSE)
}

kept_per_grna <- vapply(individual, function(idx) sum(!is.na(to_in_use[idx])), integer(1))
grna_map$n_cells <- kept_per_grna[match(grna_map$grna_id, names(kept_per_grna))]
# A gRNA in the target table with no entry in the assignment list contributes no cells rather than
# an NA that would poison its target's sum.
n_absent <- sum(is.na(grna_map$n_cells))
if (n_absent > 0) {
  cat(sprintf("  %d of %d gRNAs in @grna_target_data_frame have no assignment entry; counted as 0\n",
              n_absent, nrow(grna_map)))
  grna_map$n_cells[is.na(grna_map$n_cells)] <- 0L
}

targets <- names(union_idxs)
by_target <- split(grna_map$n_cells, grna_map$grna_target)
summed <- vapply(targets, function(t) sum(by_target[[t]]), numeric(1))
n_grnas <- vapply(targets, function(t) length(by_target[[t]]), numeric(1))
union_n <- vapply(union_idxs, length, numeric(1))

usable <- summed > 0
ratio <- union_n[usable] / summed[usable]

cat(sprintf("\n=== %s ===\n", label))
cat(sprintf("cells: %d total, %d in use (%d dropped by QC)\n",
            n_all, length(cells_in_use), n_all - length(cells_in_use)))
cat(sprintf("targets: %d | individual gRNAs: %d | gRNAs per target: median %.0f\n",
            length(targets), length(individual), median(n_grnas)))
cat(sprintf("targets with a positive summed count: %d\n", sum(usable)))
cat("\nunion / sum over the target's gRNAs:\n")
q <- quantile(ratio, c(0, 0.01, 0.25, 0.5, 0.75, 0.99, 1))
for (i in seq_along(q)) {
  cat(sprintf("  %-5s %.4f\n", paste0(names(q)[i]), q[[i]]))
}
cat(sprintf("\n  mean %.4f | share of targets where union == sum: %.1f%%\n",
            mean(ratio), 100 * mean(abs(ratio - 1) < 1e-9)))
cat(sprintf("  substituting the union for the sum would change num_trt_cells by a median of %.1f%%\n",
            100 * (median(ratio) - 1)))
