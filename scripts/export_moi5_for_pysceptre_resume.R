#!/usr/bin/env Rscript
# Resumes from the already-rebuilt post-QC moi5 sceptre object (so_postqc.rds)
# and exports what pysceptre needs, subsetting the response matrix to only
# the genes that actually appear in pass_qc pairs (244 of 38,606) rather than
# densifying the whole thing (which was measured to need ~37.7 GiB and killed
# the previous run).

suppressPackageStartupMessages({
  library(sceptre)
  library(Matrix)
})
source("/mnt/disks/sw-dev-disk/wtc-11/element-gene-power-analysis/lib/sceptre_io.R")

OUT_DIR <- "/mnt/disks/sw-dev-disk/pysceptre/tests/validation/moi5_real"
RESULTS_CRT_FP <- "/mnt/disks/sw-dev-disk/wtc-11/resampling-mechanism-comparison-moi5/results_crt.rds"

cat("Loading rebuilt post-QC object...\n")
so <- readRDS(file.path(OUT_DIR, "so_postqc.rds"))
cells_in_use <- so@cells_in_use

cat("Computing pass_qc pairs...\n")
pairs_passing <- qc_passing_pairs(so)
write.csv(pairs_passing, file.path(OUT_DIR, "pairs.csv"), row.names = FALSE)
cat("  ", nrow(pairs_passing), "pass_qc pairs,", length(unique(pairs_passing$response_id)),
    "distinct genes,", length(unique(pairs_passing$grna_target)), "distinct targets\n")

genes_needed <- unique(pairs_passing$response_id)

cat("Exporting response_matrix (subset to genes actually tested)...\n")
response_matrix_full <- sceptre:::get_response_matrix(so)  # all 38,606 genes x all cells
response_matrix <- response_matrix_full[genes_needed, cells_in_use, drop = FALSE]
rm(response_matrix_full)
gc()
response_matrix_dense <- as.matrix(response_matrix)
storage.mode(response_matrix_dense) <- "double"
writeBin(as.vector(t(response_matrix_dense)), file.path(OUT_DIR, "response_matrix.bin"), size = 8)
writeLines(rownames(response_matrix_dense), file.path(OUT_DIR, "response_matrix.genes.txt"))
cat("  response_matrix:", nrow(response_matrix_dense), "genes x", ncol(response_matrix_dense), "cells\n")

cat("Exporting covariate_matrix...\n")
covariate_matrix <- so@covariate_matrix[cells_in_use, , drop = FALSE]
storage.mode(covariate_matrix) <- "double"
writeBin(as.vector(t(covariate_matrix)), file.path(OUT_DIR, "covariate_matrix.bin"), size = 8)
writeLines(colnames(covariate_matrix), file.path(OUT_DIR, "covariate_matrix.cols.txt"))
cat("  covariate_matrix:", nrow(covariate_matrix), "cells x", ncol(covariate_matrix), "covariates:",
    paste(colnames(covariate_matrix), collapse = ", "), "\n")

cat("Exporting grna_target_cells (union target -> treated cell indices, already cells_in_use-relative)...\n")
grna_group_idxs <- so@grna_assignments$grna_group_idxs
targets_needed <- unique(pairs_passing$grna_target)
target_rows <- do.call(rbind, lapply(targets_needed, function(target) {
  idxs <- grna_group_idxs[[target]]
  if (is.null(idxs) || length(idxs) == 0) return(NULL)
  data.frame(grna_target = target, cell_index_0based = as.integer(idxs) - 1L)
}))
write.csv(target_rows, file.path(OUT_DIR, "grna_target_cells.csv"), row.names = FALSE)
cat("  ", length(targets_needed), "targets,", nrow(target_rows), "target-cell membership rows\n")

cat("Exporting R's real discovery_result for comparison...\n")
r_results <- readRDS(RESULTS_CRT_FP)
write.csv(r_results$discovery_result, file.path(OUT_DIR, "r_discovery_result.csv"), row.names = FALSE)
cat("  ", nrow(r_results$discovery_result), "rows\n")

cat("DONE. Exported to", OUT_DIR, "\n")
