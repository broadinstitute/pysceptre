# Shared extraction logic: get an in-memory post-QC sceptre object out of R.
#
# This writes a columnar intermediate, which scripts/make_h5mu.py converts to
# the .h5mu pysceptre actually reads. R is involved only to extract a dataset
# from ondisc once; it is not part of pysceptre's pipeline.
#
# Kept separate from the CLI so the benchmark scripts can export the *exact*
# object it is about to analyze, rather than re-deriving inputs and hoping they
# match. They do not: re-running assign_grnas(thresholding, threshold = 5) on
# one real object reproduced only 39 of 2,974 targets' cell sets, which is
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
# scripts/make_h5mu.py converts them and deletes them.
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
  # as.numeric first: 38,606 genes x 131,055 cells overflows R's 32-bit integer
  # multiply and silently yields NA.
  density <- n_entries / (as.numeric(length(gene_ids)) * as.numeric(n_cells))
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

  # gRNA assignments as ONE matrix with an annotation frame, rather than a
  # table per kind. Columns are "assignment units"; the annotation says what
  # each unit is, exactly as a var frame annotates a matrix's columns.
  #
  # The two kinds exist because sceptre stores two, and only two:
  #
  #   grna_group_idxs     one entry per TARGET, the union of that target's
  #                       gRNAs. "non-targeting" is deliberately absent -- 2,974
  #                       keys against 2,975 distinct targets on one real screen.
  #   indiv_nt_grna_idxs  one entry per individual NON-TARGETING gRNA (1,499).
  #
  # So per-gRNA resolution exists for NTCs and nowhere else; sceptre keeps it
  # precisely because the calibration check regroups individual NTC gRNAs into
  # synthetic targets, and discards it for targeting gRNAs, which are only ever
  # used as a union. Exporting the union table alone therefore drops every NTC
  # -- not by oversight, but because "non-targeting" is not a key in it.
  #
  # Both kinds are already relative to cells_in_use and 1-based, and both go
  # through this one code path: re-deriving assignments independently is what
  # once reproduced only 39 of 2,974 targets.
  cat("Exporting gRNA assignments...\n")
  grna_group_idxs <- so@grna_assignments$grna_group_idxs
  indiv_nt <- so@grna_assignments$indiv_nt_grna_idxs

  as_rows <- function(idx_list) {
    if (is.null(idx_list) || length(idx_list) == 0) return(NULL)
    do.call(rbind, lapply(names(idx_list), function(unit) {
      idxs <- idx_list[[unit]]
      if (length(idxs) == 0) return(NULL)
      data.frame(unit_id = unit, cell_index = as.integer(idxs) - 1L)
    }))
  }
  assignment_rows <- rbind(as_rows(grna_group_idxs), as_rows(indiv_nt))
  if (max(assignment_rows$cell_index) >= n_cells) {
    stop("gRNA cell indices exceed the cells_in_use count -- indexing convention changed",
         call. = FALSE)
  }
  write_parquet(assignment_rows, file.path(out_dir, "grna_assignments.parquet"))

  annotation <- rbind(
    data.frame(unit_id = names(grna_group_idxs),
               grna_target = names(grna_group_idxs),
               unit_kind = "target"),
    if (!is.null(indiv_nt) && length(indiv_nt) > 0)
      data.frame(unit_id = names(indiv_nt),
                 grna_target = "non-targeting",
                 unit_kind = "ntc_grna")
  )
  write_parquet(annotation, file.path(out_dir, "grna_annotation.parquet"))
  cat("   ", length(grna_group_idxs), "targets +", length(indiv_nt), "NTC gRNAs =",
      nrow(annotation), "units,", nrow(assignment_rows), "membership rows\n")
  if (is.null(indiv_nt) || length(indiv_nt) == 0) {
    cat("    no individual NTC gRNAs; the calibration check is not runnable from this export\n")
  }

  # The calibration check's pairs and results, when the object has been through
  # run_calibration_check. Worth carrying because R's pair selection is
  # unseeded and so cannot be reproduced by re-running: these are the only
  # record of which pairs a given result was computed on, and without them a
  # pair-by-pair comparison against that result is impossible.
  ncp <- so@negative_control_pairs
  if (!is.null(ncp) && nrow(ncp) > 0) {
    # R names the column grna_group here but grna_target everywhere else.
    if ("grna_group" %in% names(ncp)) names(ncp)[names(ncp) == "grna_group"] <- "grna_target"
    write_parquet(as.data.frame(ncp), file.path(out_dir, "negative_control_pairs.parquet"))
    cat("   R negative_control_pairs:", nrow(ncp), "rows\n")
  }
  calibration_result <- so@calibration_result
  if (!is.null(calibration_result) && nrow(calibration_result) > 0) {
    cr <- as.data.frame(calibration_result)
    if ("grna_group" %in% names(cr)) names(cr)[names(cr) == "grna_group"] <- "grna_target"
    write_parquet(cr, file.path(out_dir, "calibration_result.parquet"))
    cat("   R calibration_result:", nrow(cr), "rows\n")
  }

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
    # Pairwise QC thresholds. The calibration check needs these because it
    # samples only (gene, synthetic-target) pairs that already pass them --
    # unlike discovery, where the pairs are given and failures are reported
    # in-band with pass_qc = FALSE.
    n_nonzero_trt_thresh = so@n_nonzero_trt_thresh,
    n_nonzero_cntrl_thresh = so@n_nonzero_cntrl_thresh,
    n_ok_discovery_pairs = so@n_ok_discovery_pairs,
    # R's `p_hat` in construct_negative_control_pairs_v2: the fraction of
    # discovery pairs clearing pairwise QC, which sizes how many synthetic
    # negative-control groups get built. pysceptre cannot recompute it, because
    # it is only ever handed pairs that already passed, so it is recorded here.
    # Confirmed load-bearing on day0: R built 625 groups, and the rule
    # reproduces 625 only with this rate (0.9571); assuming 1.0 gives 598.
    discovery_pass_qc_rate = if (!is.null(so@discovery_pairs_with_info) &&
                                 nrow(so@discovery_pairs_with_info) > 0)
      mean(so@discovery_pairs_with_info$pass_qc) else NA_real_,
    # R's default group size: median gRNAs per real target, capped at the
    # number of NTC gRNAs available.
    calibration_group_size = sceptre:::compute_calibration_group_size(so@grna_target_data_frame),
    n_ntc_grnas = length(so@grna_assignments$indiv_nt_grna_idxs),
    formula = paste(deparse(so@formula_object), collapse = " "),
    sceptre_version = as.character(packageVersion("sceptre")),
    exported_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
    source_object = source_label
  )
  write_json(metadata, file.path(out_dir, "metadata.json"),
             auto_unbox = TRUE, pretty = TRUE)

  invisible(out_dir)
}
