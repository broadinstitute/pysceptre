#!/usr/bin/env Rscript
# Rebuilds the post-QC moi5 sceptre object (same recipe as
# resampling-mechanism-comparison-moi5/common.R: thresholding assign_grnas,
# threshold=15, union strategy, side="left"), then exports everything
# pysceptre's run_discovery_analysis needs as plain binary/CSV files:
#   response_matrix.bin   (n_genes x n_cells, dense, float64, row-major)
#   response_matrix.meta  (gene_ids, one per line)
#   covariate_matrix.bin  (n_cells x p, dense, float64, row-major)
#   covariate_matrix.meta (column names, one per line)
#   grna_target_cells.csv (grna_target, cell_index_0based)
#   pairs.csv             (grna_target, response_id) -- pass_qc == TRUE only
#   r_discovery_result.csv (R's own actual discovery_result, for comparison)

suppressPackageStartupMessages({
  library(sceptre)
  library(ondisc)
  library(Matrix)
})

source("/mnt/disks/sw-dev-disk/wtc-11/element-gene-power-analysis/lib/sceptre_io.R")

BASE <- "/mnt/disks/sw-dev-disk/wtc-11"
SO_PATH <- file.path(BASE, "grna-single-target-moi5-ondisc/sceptre_object.rds")
GENE_ODM_FP <- file.path(BASE, "grna-single-target-moi5-ondisc/gene.odm")
GRNA_ODM_FP <- file.path(BASE, "grna-single-target-moi5-ondisc/grna.odm")
DISCOVERY_PAIRS_FP <- file.path(BASE, "discovery_pairs.rds")
POSITIVE_PAIRS_FP <- file.path(BASE, "positive.rds")
RESULTS_CRT_FP <- file.path(BASE, "resampling-mechanism-comparison-moi5/results_crt.rds")

OUT_DIR <- "/mnt/disks/sw-dev-disk/pysceptre/tests/validation/moi5_real"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

t_start <- Sys.time()
cat("Reattaching odm-backed matrices...\n")
so <- sceptre:::read_ondisc_backed_sceptre_object(
  sceptre_object_fp = SO_PATH,
  response_odm_file_fp = GENE_ODM_FP,
  grna_odm_file_fp = GRNA_ODM_FP
)

discovery_pairs <- readRDS(DISCOVERY_PAIRS_FP)
positive_control_pairs <- readRDS(POSITIVE_PAIRS_FP)

cat("Materializing odm response matrix into memory...\n")
t0 <- Sys.time()
dense_response <- materialize_odm_response_matrix(so@response_matrix[[1]])
so <- sceptre:::set_response_matrix(so, dense_response)
cat("  took", format(Sys.time() - t0), "\n")

cat("set_analysis_parameters()...\n")
so <- sceptre::set_analysis_parameters(
  so,
  discovery_pairs = discovery_pairs,
  positive_control_pairs = positive_control_pairs,
  side = "left",
  grna_integration_strategy = "union",
  resampling_mechanism = "crt"
)

cat("assign_grnas(thresholding, threshold=15)...\n")
t0 <- Sys.time()
so <- sceptre::assign_grnas(so, method = "thresholding", threshold = 15,
                             parallel = TRUE, n_processors = 7)
cat("  took", format(Sys.time() - t0), "\n")

cat("run_qc()...\n")
t0 <- Sys.time()
so <- sceptre::run_qc(so)
cat("  took", format(Sys.time() - t0), "\n")

saveRDS(so, file.path(OUT_DIR, "so_postqc.rds"))
cat("Saved rebuilt post-QC object to", file.path(OUT_DIR, "so_postqc.rds"), "\n")

# ---- export everything pysceptre needs ----
cat("Exporting response_matrix...\n")
response_matrix <- sceptre:::get_response_matrix(so)  # genes x all cells
cells_in_use <- so@cells_in_use  # 1-based indices into the full cell universe
response_matrix <- response_matrix[, cells_in_use, drop = FALSE]
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

cat("Exporting grna_target_cells (union target -> treated cell indices)...\n")
grna_group_idxs <- so@grna_assignments$grna_group_idxs  # 1-based indices into cells_in_use-space already
# cross-check: are these indices already relative to cells_in_use, or the full cell universe?
# sceptre's internal convention is that grna_group_idxs indexes into the *cells_in_use* ordering
# (verified via get_idx_vector_factory/discovery_ntcells_crt usage patterns read earlier), so no
# further remapping against cells_in_use is needed here -- just convert 1-based -> 0-based.
target_rows <- do.call(rbind, lapply(names(grna_group_idxs), function(target) {
  idxs <- grna_group_idxs[[target]]
  if (length(idxs) == 0) return(NULL)
  data.frame(grna_target = target, cell_index_0based = as.integer(idxs) - 1L)
}))
write.csv(target_rows, file.path(OUT_DIR, "grna_target_cells.csv"), row.names = FALSE)
cat("  ", length(grna_group_idxs), "targets,", nrow(target_rows), "target-cell membership rows\n")

cat("Exporting pass_qc pairs...\n")
pairs_passing <- qc_passing_pairs(so)
write.csv(pairs_passing, file.path(OUT_DIR, "pairs.csv"), row.names = FALSE)
cat("  ", nrow(pairs_passing), "pass_qc pairs\n")

cat("Exporting R's real discovery_result for comparison...\n")
r_results <- readRDS(RESULTS_CRT_FP)
write.csv(r_results$discovery_result, file.path(OUT_DIR, "r_discovery_result.csv"), row.names = FALSE)
cat("  ", nrow(r_results$discovery_result), "rows\n")

cat("\nTotal time:", format(Sys.time() - t_start), "\n")
cat("DONE. Exported to", OUT_DIR, "\n")
