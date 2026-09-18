# Shared extraction logic: get an in-memory post-QC sceptre object out of R.
#
# This writes a columnar intermediate, which scripts/make_h5ad.py converts to
# the h5ad pysceptre actually reads. R is involved only to extract a dataset
# from ondisc once; it is not part of pysceptre's pipeline.
#
# Kept separate from the CLI so the benchmark scripts can export the *exact*
# object it is about to analyze, rather than re-deriving inputs and hoping they
# match. They do not: re-running assign_grnas(thresholding, threshold = 5) on
# the moi5 object reproduced only 39 of 2,974 targets' cell sets, which is
# enough to break fold-change agreement (a deterministic quantity) from ~1e-15
# to 0.67.

suppressPackageStartupMessages({
  library(sceptre)
  library(arrow)
  library(jsonlite)
  library(Matrix)
})

qc_passing_pairs <- function(so) {
  pairs <- so@discovery_pairs_with_info
  if (nrow(pairs) == 0) stop("object has no discovery pairs; run run_qc() first", call. = FALSE)
  keep <- pairs[pairs$pass_qc, c("response_id", "grna_group")]
  data.frame(
    response_id = as.character(keep$response_id),
    grna_target = as.character(keep$grna_group),
    stringsAsFactors = FALSE
  )
}

# The .parquet files written here are an intermediate, not the dataset:
# scripts/make_h5ad.py converts them and deletes them.
export_sceptre_object <- function(so, out_dir, all_genes = FALSE,
                                  source_label = "<in-memory>") {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  cells_in_use <- so@cells_in_use
  n_cells <- length(cells_in_use)

  pairs <- qc_passing_pairs(so)
  write_parquet(pairs, file.path(out_dir, "pairs.parquet"))
  cat("  pairs:", nrow(pairs), "passing QC\n")

  response_matrix <- sceptre:::get_response_matrix(so)
  all_gene_ids <- rownames(response_matrix)
  gene_ids <- if (all_genes) all_gene_ids else sort(unique(pairs$response_id))
  missing <- setdiff(gene_ids, all_gene_ids)
  if (length(missing)) {
    stop(sprintf("%d pair genes absent from the response matrix, e.g. %s",
                 length(missing), missing[[1]]), call. = FALSE)
  }
  gene_rows <- match(gene_ids, all_gene_ids)

  cat("Exporting response matrix as sparse triplets:", length(gene_ids), "genes x",
      n_cells, "cells...\n")
  t0 <- Sys.time()
  chunks <- vector("list", length(gene_rows))
  n_entries <- 0
  for (k in seq_along(gene_rows)) {
    # One odm row at a time: never materialise the full matrix.
    counts <- response_matrix[gene_rows[[k]], ][cells_in_use]
    nz <- which(counts != 0)
    n_entries <- n_entries + length(nz)
    chunks[[k]] <- data.frame(
      gene_index = rep.int(as.integer(k - 1L), length(nz)),
      cell_index = as.integer(nz - 1L),
      value = as.double(counts[nz])
    )
    if (k %% 50 == 0) cat("   ", k, "/", length(gene_rows), "\n")
  }
  triplets <- do.call(rbind, chunks)
  write_parquet(triplets, file.path(out_dir, "response_matrix.parquet"))
  density <- n_entries / (length(gene_ids) * n_cells)
  cat("   ", format(n_entries, big.mark = ","), "nonzero entries",
      sprintf("(%.1f%% dense)", 100 * density), "in", format(Sys.time() - t0), "\n")

  write_parquet(
    data.frame(gene_index = seq_along(gene_ids) - 1L, response_id = gene_ids),
    file.path(out_dir, "gene_ids.parquet")
  )

  cat("Exporting covariate matrix...\n")
  covariate_matrix <- so@covariate_matrix[cells_in_use, , drop = FALSE]
  covariate_df <- as.data.frame(covariate_matrix)
  names(covariate_df) <- colnames(covariate_matrix)
  write_parquet(covariate_df, file.path(out_dir, "covariate_matrix.parquet"))
  cat("   ", nrow(covariate_df), "cells x", ncol(covariate_df), "covariates\n")

  cat("Exporting gRNA target -> treated cells...\n")
  # sceptre stores these already relative to cells_in_use, 1-based.
  grna_group_idxs <- so@grna_assignments$grna_group_idxs
  target_rows <- do.call(rbind, lapply(names(grna_group_idxs), function(target) {
    idxs <- grna_group_idxs[[target]]
    if (length(idxs) == 0) return(NULL)
    data.frame(grna_target = target, cell_index = as.integer(idxs) - 1L)
  }))
  if (max(target_rows$cell_index) >= n_cells) {
    stop("gRNA cell indices exceed the cells_in_use count -- indexing convention changed",
         call. = FALSE)
  }
  write_parquet(target_rows, file.path(out_dir, "grna_target_cells.parquet"))
  cat("   ", length(grna_group_idxs), "targets,", nrow(target_rows), "membership rows\n")

  discovery_result <- so@discovery_result
  if (!is.null(discovery_result) && nrow(discovery_result) > 0) {
    write_parquet(as.data.frame(discovery_result),
                  file.path(out_dir, "discovery_result.parquet"))
    cat("   R discovery_result:", nrow(discovery_result), "rows\n")
  } else {
    cat("   no R discovery_result present (pass --discovery-result to include one)\n")
  }

  metadata <- list(
    n_cells = n_cells,
    n_genes = length(gene_ids),
    n_covariates = ncol(covariate_df),
    covariate_names = colnames(covariate_matrix),
    n_targets = length(grna_group_idxs),
    n_pairs = nrow(pairs),
    n_nonzero = n_entries,
    all_genes = all_genes,
    # Analysis parameters, so a pysceptre run can be set up to match.
    side_code = so@side_code,
    resampling_approximation = so@resampling_approximation,
    grna_integration_strategy = so@grna_integration_strategy,
    run_permutations = so@run_permutations,
    control_group_complement = so@control_group_complement,
    multiple_testing_alpha = so@multiple_testing_alpha,
    B1 = so@B1, B2 = so@B2, B3 = so@B3,
    formula = paste(deparse(so@formula_object), collapse = " "),
    sceptre_version = as.character(packageVersion("sceptre")),
    exported_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
    source_object = source_label
  )
  write_json(metadata, file.path(out_dir, "metadata.json"),
             auto_unbox = TRUE, pretty = TRUE)

  invisible(out_dir)
}
