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

slot_or_null <- function(object, name) {
  if (methods::.hasSlot(object, name)) methods::slot(object, name) else NULL
}

# Each NT gRNA's cells as cells_in_use positions, whichever control group the object uses.
#
# With the complement control group sceptre already stores them that way. With nt_cells,
# update_indiv_grna_assignments_for_nt_cells() (R/cellwise_qc_functs.R, sceptre 0.10.3) replaces
# them with positions WITHIN grna_assignments$all_nt_idxs: gRNA i becomes start_i:stop_i, and
# all_nt_idxs holds the cells. Written out unconverted, those positions read as cell indices.
#
# Both cases are checked against the pre-QC assignment mapped through cells_in_use, which reaches
# the same cells without going through either encoding.
nt_grna_cells_in_use <- function(so, to_in_use) {
  indiv_nt <- so@grna_assignments$indiv_nt_grna_idxs
  all_nt_idxs <- so@grna_assignments$all_nt_idxs
  if (length(indiv_nt) == 0) return(indiv_nt)
  if (!is.null(all_nt_idxs)) {
    positions <- as.integer(unlist(indiv_nt, use.names = FALSE))
    # sceptre builds the positions as consecutive runs over unique(unlist(cells)), so they cover
    # 1..length(all_nt_idxs) exactly once, in order, only when the NT gRNAs' cell sets are
    # disjoint. Anything else means the decoding below would pick the wrong cells.
    if (!identical(positions, seq_along(all_nt_idxs))) {
      stop("grna_assignments$indiv_nt_grna_idxs does not partition all_nt_idxs in order (",
           length(positions), " positions, ", length(all_nt_idxs), " NT cells): either the NT ",
           "gRNAs share cells or sceptre changed its nt_cells encoding.", call. = FALSE)
    }
    indiv_nt <- lapply(indiv_nt, function(p) all_nt_idxs[p])
  } else if (!isTRUE(so@control_group_complement)) {
    stop("control_group_complement is FALSE but grna_assignments has no all_nt_idxs, so the ",
         "NT gRNA indices cannot be decoded.", call. = FALSE)
  }

  initial <- so@initial_grna_assignment_list
  checkable <- intersect(names(indiv_nt), names(initial))
  mismatched <- Filter(function(grna_id) {
    expected <- to_in_use[initial[[grna_id]]]
    !identical(sort(as.integer(expected[expected > 0L])), sort(as.integer(indiv_nt[[grna_id]])))
  }, checkable)
  if (length(mismatched) > 0) {
    stop(length(mismatched), " of ", length(checkable), " NT gRNAs do not match the pre-QC ",
         "assignment restricted to cells_in_use, including: ",
         paste(utils::head(mismatched, 3), collapse = ", "), call. = FALSE)
  }
  cat("   ", length(indiv_nt), "NT gRNAs",
      if (!is.null(all_nt_idxs)) "decoded from positions within all_nt_idxs;" else "in cell space;",
      length(checkable), "match the pre-QC assignment exactly\n")
  if (length(checkable) < length(indiv_nt)) {
    cat("    NOTE:", length(indiv_nt) - length(checkable),
        "NT gRNAs are absent from @initial_grna_assignment_list and were not cross-checked\n")
  }
  indiv_nt
}

# The .parquet files written here are an intermediate, not the dataset:
# scripts/make_h5mu.py converts them and deletes them.
# ONE CELL SPACE PER FILE. By default that is `cells_in_use`, the QC-passing cells an analysis
# runs on, and every index written -- matrix columns, covariate rows, each gRNA unit's cells -- is
# relative to it. With `all_cells = TRUE` it is instead every cell in the object, and *all* of them
# move together: the matrix keeps its QC-failed columns and the gRNA units are re-indexed to
# absolute positions to match. Mixing the two would produce a file where `counts[, target_cells]`
# silently selects the wrong cells, which no downstream number would reveal.
#
# `all_cells` exists for simulation, not analysis. DESeq2 "poscounts" size factors are a per-cell
# reduction against a per-gene geometric mean taken across the whole matrix, so which cells are in
# the matrix changes the size factors of the cells that stay -- measured on day0, a median 0.46 %
# shift in size factor and 0.36 % in normalised gene mean (max 3.0 %). WattEG's R implementation
# computes them over every cell, so reproducing it needs every cell. The loader subsets back to
# `cells_in_use` by default, so an analysis reads an `all_cells` file exactly as it reads any other.
export_sceptre_object <- function(so, out_dir, all_genes = TRUE, all_cells = TRUE,
                                  source_label = "<in-memory>") {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  cells_in_use <- so@cells_in_use

  pairs <- qc_passing_pairs(so)
  write_parquet(pairs, file.path(out_dir, "pairs.parquet"))
  cat("  pairs:", nrow(pairs), "passing QC\n")

  response_matrix <- sceptre:::get_response_matrix(so)
  # Resolved here, once, because everything below indexes through it.
  cells_kept <- if (all_cells) seq_len(ncol(response_matrix)) else cells_in_use
  n_cells <- length(cells_kept)
  if (all_cells) {
    cat("Exporting ALL", n_cells, "cells;", length(cells_in_use), "of them pass QC\n")
  }
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
    counts <- response_matrix[gene_rows[[k]], ][cells_kept]
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
  # @covariate_matrix is indexed by absolute cell position and carries every cell, which is why
  # the default path subsets it here rather than reading a pre-subset slot.
  covariate_matrix <- so@covariate_matrix[cells_kept, , drop = FALSE]
  covariate_df <- as.data.frame(covariate_matrix)
  names(covariate_df) <- colnames(covariate_matrix)
  write_parquet(covariate_df, file.path(out_dir, "covariate_matrix.parquet"))
  cat("   ", nrow(covariate_df), "cells x", ncol(covariate_df), "covariates\n")

  # Which of the exported cells passed QC. Written unconditionally, all TRUE under the default,
  # so a reader has one thing to look at rather than a column that may or may not be there.
  in_use <- rep(all_cells == FALSE, n_cells)
  if (all_cells) in_use[cells_in_use] <- TRUE
  write_parquet(data.frame(in_use = in_use), file.path(out_dir, "cell_annotation.parquet"))

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
  # Both kinds are relative to cells_in_use and 1-based once the NT gRNAs are decoded (see
  # nt_grna_cells_in_use: an nt_cells object stores them as positions in its NT-cell pool), and
  # both go through this one code path: re-deriving assignments independently is what once
  # reproduced only 39 of 2,974 targets.
  cat("Exporting gRNA assignments...\n")
  # Both slots index positions *within cells_in_use*. Under `all_cells` the file's space is
  # absolute, so they are mapped out to it -- the one place the target and NTC units move, and they
  # move together with the matrix rather than against it.
  # Both slots stay in their native cells_in_use space here and are converted to the file's space
  # only when the rows are built. The round-trip check further down compares them against the
  # individual guides, and that property is defined in the cells_in_use space -- converting first
  # would leave the check comparing absolute positions against relative ones.
  grna_group_idxs <- so@grna_assignments$grna_group_idxs
  # R's NT-cell pool, present only for the nt_cells control group. Its order is the order R
  # indexes the pool by position, so it is written as is, never sorted.
  all_nt_idxs <- so@grna_assignments$all_nt_idxs

  # Absolute cell position -> cells_in_use position (0 where QC removed the cell).
  n_total_cells <- ncol(response_matrix)
  to_in_use <- integer(n_total_cells)
  to_in_use[cells_in_use] <- seq_along(cells_in_use)
  # cells_in_use position -> this file's cell space, applied after the round-trip checks, which are
  # defined in the cells_in_use space where the property lives.
  to_file_cells <- if (all_cells) function(rel) cells_in_use[rel] else identity

  indiv_nt <- nt_grna_cells_in_use(so, to_in_use)

  as_rows <- function(idx_list) {
    if (is.null(idx_list) || length(idx_list) == 0) return(NULL)
    do.call(rbind, lapply(names(idx_list), function(unit) {
      idxs <- idx_list[[unit]]
      if (length(idxs) == 0) return(NULL)
      data.frame(unit_id = unit, cell_index = as.integer(to_file_cells(idxs)) - 1L)
    }))
  }
  assignment_rows <- rbind(as_rows(grna_group_idxs), as_rows(indiv_nt))
  if (max(assignment_rows$cell_index) >= n_cells) {
    stop("gRNA cell indices exceed the exported cell count -- indexing convention changed",
         call. = FALSE)
  }
  annotation <- rbind(
    data.frame(unit_id = names(grna_group_idxs),
               grna_target = names(grna_group_idxs),
               unit_kind = "target"),
    if (!is.null(indiv_nt) && length(indiv_nt) > 0)
      data.frame(unit_id = names(indiv_nt),
                 grna_target = "non-targeting",
                 unit_kind = "ntc_grna")
  )
  # --- individual TARGETING gRNAs ----------------------------------------------------------
  #
  # A third `unit_kind`, and the only per-gRNA resolution sceptre keeps for targeting guides:
  # @initial_grna_assignment_list, keyed by grna_id, covering every gRNA whether targeting or not.
  # @grna_assignments (above) keeps only the per-target union, because the union is all a discovery
  # analysis ever uses.
  #
  # A *simulation* needs the guides themselves. WattEG draws a separate effect size per gRNA around
  # the target's effect size -- guide-to-guide variability, sd 0.13 -- so it has to know which cells
  # carry which guide, not merely which carry the target. Nothing in pysceptre's own analysis path
  # reads these units; `load_h5mu` selects units by kind, so they are inert for every existing
  # reader.
  #
  # TWO INDEX SPACES MEET HERE, and this is the only place in the export where they do.
  # @initial_grna_assignment_list holds *absolute* cell positions (up to 586,309 on day0) while
  # everything written here -- matrix columns, covariate rows, grna_group_idxs -- is relative to
  # cells_in_use (567,690). The map below sends the first into the second and drops the cells QC
  # removed. The drop is counted rather than assumed harmless: silently losing a guide's cells
  # would look exactly like a guide that had fewer cells to begin with.
  cat("Exporting individual targeting gRNAs...\n")
  initial <- so@initial_grna_assignment_list
  gtdf <- so@grna_target_data_frame

  # A NAMED VECTOR, NOT A LOOKUP TABLE, because the gRNA -> target map is many-to-many: on day0,
  # 1,673 of 43,736 guides sit inside two or three overlapping candidate elements and so appear
  # under two or three targets (45,463 rows against 43,736 distinct ids). R keeps the repeats;
  # anything that collapses them -- a dict, a match() -- silently drops those guides from every
  # target but one, which would leave 216 of 3,071 targets simulating with an incomplete guide set.
  # Every use below either intersects against this vector's names or splits on it, so the repeats
  # survive.
  target_of <- stats::setNames(as.character(gtdf$grna_target), as.character(gtdf$grna_id))
  targets_per_grna <- tapply(as.character(gtdf$grna_target), as.character(gtdf$grna_id),
                             function(x) length(unique(x)))

  # gRNA MEMBERSHIP IS ALWAYS POST-QC, in both cell spaces, and that is a statement about sceptre
  # rather than a convenience. @grna_assignments$grna_group_idxs -- the target unions, and what an
  # analysis uses -- is built after QC and indexes only cells_in_use.
  # @initial_grna_assignment_list is the *input*, recorded before QC ran, so a target's guides
  # between them cover cells the target itself does not: on day0's first target, 518 cells against
  # 493, the 25 extra being cells QC removed.
  #
  # Restricting the guides to cells_in_use is what makes "a target's cells are the union of its
  # guides' cells" true in every file, which the assertion below then checks and a simulation then
  # relies on. Carrying the pre-QC surplus instead would buy nothing -- the cells have no
  # covariates, and no test sees them -- at the price of an invariant.
  #
  # So `--all-cells` adds cells to the EXPRESSION side only: more matrix columns, for the
  # whole-matrix geometric means poscounts needs. It adds no gRNA memberships.
  # (`to_in_use` and `to_file_cells` are built above, where the NT gRNAs needed them first.)
  targeting_ids <- setdiff(unique(names(initial)),
                           unique(names(target_of)[target_of == "non-targeting"]))
  targeting_ids <- targeting_ids[targeting_ids %in% names(target_of)]
  n_dropped <- 0L
  targeting_rows <- do.call(rbind, lapply(targeting_ids, function(grna_id) {
    abs_cells <- initial[[grna_id]]
    if (length(abs_cells) == 0) return(NULL)
    if (max(abs_cells) > n_total_cells) {
      stop("@initial_grna_assignment_list holds cell index ", max(abs_cells), " but the response ",
           "matrix has ", n_total_cells, " columns; these were assumed to be absolute positions.",
           call. = FALSE)
    }
    mapped <- to_in_use[abs_cells]
    n_dropped <<- n_dropped + sum(mapped == 0L)
    mapped <- sort(mapped[mapped > 0L])
    if (length(mapped) == 0) return(NULL)
    data.frame(unit_id = grna_id, cell_index = as.integer(to_file_cells(mapped)) - 1L)
  }))

  # The check that matters. Unioning a target's individual guides must reproduce the union sceptre
  # stored for that target: if it does not, a simulation's guide-level perturbation status
  # disagrees with the target-level status the very same simulation is tested against -- silently,
  # and in a way no downstream number would reveal. Exhaustive over every target rather than
  # sampled; it is seconds on 3,071 targets, and the object it would catch is by definition the
  # unusual one.
  #
  # Hard for `union` integration, which is what the property is *about* and what a power simulation
  # requires. Under any other strategy a target's cells are not the union of its guides' cells by
  # construction, so a mismatch is information rather than corruption, and it is reported as such.
  by_target <- split(names(target_of), unname(target_of))
  mismatched <- character(0)
  for (target_id in names(grna_group_idxs)) {
    guides <- intersect(by_target[[target_id]], names(initial))
    mapped <- to_in_use[unlist(initial[guides], use.names = FALSE)]
    mapped <- sort(unique(mapped[mapped > 0L]))
    if (!identical(as.integer(mapped), as.integer(sort(grna_group_idxs[[target_id]])))) {
      mismatched <- c(mismatched, target_id)
    }
  }
  if (length(mismatched) > 0) {
    msg <- sprintf(paste("%d of %d targets' cell sets are not the union of their individual gRNAs'",
                         "cell sets, including: %s"),
                   length(mismatched), length(grna_group_idxs),
                   paste(utils::head(mismatched, 3), collapse = ", "))
    if (identical(so@grna_integration_strategy, "union")) {
      stop(msg, ". The object says grna_integration_strategy = \"union\", so this should be ",
           "impossible; the export would be internally inconsistent and a power simulation run ",
           "from it would be wrong.", call. = FALSE)
    }
    cat("    NOTE:", msg, "\n")
    cat("    grna_integration_strategy is", so@grna_integration_strategy,
        "-- expected, but a simulation needing the union property cannot use this export\n")
  }

  # The annotation frame has one row per unit, so a guide with several targets cannot carry them
  # all: `grna_target` is a single value by construction. Rather than pick one of two and let a
  # reader believe it, such a guide is labelled "<multiple>" and the many-to-many truth lives in
  # grna_target_data_frame.parquet, which is written whole for exactly this reason.
  if (!is.null(targeting_rows)) {
    kept_ids <- unique(as.character(targeting_rows$unit_id))
    n_targets_of <- as.integer(targets_per_grna[kept_ids])
    single <- vapply(kept_ids, function(id) unname(target_of[names(target_of) == id])[1],
                     character(1))
    assignment_rows <- rbind(assignment_rows, targeting_rows)
    annotation <- rbind(
      annotation,
      data.frame(unit_id = kept_ids,
                 grna_target = ifelse(n_targets_of > 1L, "<multiple>", single),
                 unit_kind = "targeting_grna")
    )
    n_multi <- sum(n_targets_of > 1L)
  } else {
    kept_ids <- character(0)
    n_multi <- 0L
  }
  n_targeting_units <- if (is.null(targeting_rows)) 0L else length(unique(targeting_rows$unit_id))
  cat("   ", n_targeting_units, "targeting gRNAs,",
      format(if (is.null(targeting_rows)) 0L else nrow(targeting_rows), big.mark = ","),
      "membership rows\n")
  if (n_multi > 0) {
    cat("    ", n_multi, "guides target more than one element; annotated \"<multiple>\",",
        "see grna_target_data_frame.parquet\n")
  }
  if (n_dropped > 0) {
    cat("    ", format(n_dropped, big.mark = ","),
        "pre-QC gRNA-cell memberships dropped: those cells are not in cells_in_use, and the",
        "target unions do not carry them either\n")
  }
  if (length(mismatched) == 0) {
    cat("     union round-trip: all", length(grna_group_idxs), "targets reproduced exactly\n")
  }

  # The gRNA -> target map itself. Derivable from the annotation frame above, but only for guides
  # that have cells; a screen's design includes guides that were assigned none, and a reader
  # counting guides per target needs the design, not the subset that survived.
  write_parquet(
    data.frame(grna_id = as.character(gtdf$grna_id),
               grna_target = as.character(gtdf$grna_target),
               stringsAsFactors = FALSE),
    file.path(out_dir, "grna_target_data_frame.parquet")
  )

  write_parquet(assignment_rows, file.path(out_dir, "grna_assignments.parquet"))
  write_parquet(annotation, file.path(out_dir, "grna_annotation.parquet"))
  cat("   ", length(grna_group_idxs), "targets +", length(indiv_nt), "NTC gRNAs +",
      n_targeting_units, "targeting gRNAs =", nrow(annotation), "units,",
      format(nrow(assignment_rows), big.mark = ","), "membership rows\n")
  if (is.null(indiv_nt) || length(indiv_nt) == 0) {
    cat("    no individual NTC gRNAs; the calibration check is not runnable from this export\n")
  }

  # Raw gRNA UMI counts, every gRNA, in this file's cell space. The units above are 0/1; the dose
  # test weights each cell by its count, so the counts travel too. Read one odm row at a time for
  # an ondisc-backed object, like the response matrix.
  cat("Exporting gRNA UMI counts...\n")
  grna_matrix <- sceptre:::get_grna_matrix(so)
  grna_ids <- rownames(grna_matrix)
  if (inherits(grna_matrix, "Matrix")) {
    gm <- as(grna_matrix[, cells_kept, drop = FALSE], "TsparseMatrix")
    if (any(gm@x != round(gm@x))) stop("the gRNA matrix holds non-integer values", call. = FALSE)
    grna_triplets <- data.frame(grna_index = gm@i, cell_index = gm@j, value = as.integer(gm@x))
  } else {
    grna_triplets <- do.call(rbind, lapply(seq_along(grna_ids), function(k) {
      counts <- grna_matrix[k, ][cells_kept]
      nz <- which(counts != 0)
      data.frame(grna_index = rep.int(as.integer(k - 1L), length(nz)),
                 cell_index = as.integer(nz - 1L), value = as.integer(counts[nz]))
    }))
  }
  write_parquet(grna_triplets, file.path(out_dir, "grna_counts.parquet"))
  write_parquet(data.frame(grna_index = seq_along(grna_ids) - 1L, grna_id = grna_ids),
                file.path(out_dir, "grna_count_ids.parquet"))
  cat("   ", length(grna_ids), "gRNAs,", format(nrow(grna_triplets), big.mark = ","),
      "nonzero counts\n")
  # The nt_cells pool in R's order, 0-based in this file's cell space. It equals the NT units'
  # cells concatenated in unit order (nt_grna_cells_in_use checked the partition that implies).
  if (!is.null(all_nt_idxs)) {
    write_parquet(data.frame(cell_index = as.integer(to_file_cells(all_nt_idxs)) - 1L),
                  file.path(out_dir, "all_nt_idxs.parquet"))
    cat("    all_nt_idxs:", length(all_nt_idxs), "NT cells, in R's order\n")
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

  # The power check's pairs and results, when the object has been through
  # run_power_check. Carried for the same reason as the calibration pairs:
  # positive controls are a claim the experiment makes, and the name-matching
  # rule that sceptre falls back to finds nothing when targets are genomic
  # intervals rather than gene names -- 0 of 3,071 on this kind of screen.
  # Without these there is no way to know which target was meant to perturb
  # which gene.
  pcp <- so@positive_control_pairs_with_info
  if (!is.null(pcp) && nrow(pcp) > 0) {
    pcp <- as.data.frame(pcp)
    if ("grna_group" %in% names(pcp)) names(pcp)[names(pcp) == "grna_group"] <- "grna_target"
    write_parquet(pcp, file.path(out_dir, "positive_control_pairs.parquet"))
    cat("   R positive_control_pairs:", nrow(pcp), "rows\n")
  }
  power_result <- so@power_result
  if (!is.null(power_result) && nrow(power_result) > 0) {
    pw <- as.data.frame(power_result)
    if ("grna_group" %in% names(pw)) names(pw)[names(pw) == "grna_group"] <- "grna_target"
    write_parquet(pw, file.path(out_dir, "power_result.parquet"))
    cat("   R power_result:", nrow(pw), "rows\n")
  }

  # The discovery pairs *with* their QC columns, not just the passing ids. `pairs.parquet` above
  # is the analysis input -- the pairs an engine is handed -- and deliberately carries only the two
  # id columns. This is the record of the screen: `n_nonzero_trt`, `n_nonzero_cntrl` and `pass_qc`
  # are per-pair facts about the real data, constant across any number of simulated replicates, and
  # they are the first thing anyone reads when a pair's power comes out surprising. There is
  # nowhere else to recover them from once the object is gone.
  dpwi <- so@discovery_pairs_with_info
  if (!is.null(dpwi) && nrow(dpwi) > 0) {
    dpwi <- as.data.frame(dpwi)
    if ("grna_group" %in% names(dpwi)) names(dpwi)[names(dpwi) == "grna_group"] <- "grna_target"
    write_parquet(dpwi, file.path(out_dir, "discovery_pairs_with_info.parquet"))
    cat("   discovery_pairs_with_info:", nrow(dpwi), "rows,",
        sum(dpwi$pass_qc), "passing QC\n")
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
    # Per-gRNA targeting units: present only in exports written after WattEG's simulation support.
    # A reader that needs them should check this rather than discovering an empty selection.
    n_targeting_grnas = n_targeting_units,
    n_grna_cells_dropped_by_qc = n_dropped,
    # Guides that belong to more than one target. Their annotation row says "<multiple>";
    # grna_target_data_frame is the map that answers which ones.
    n_multi_target_grnas = n_multi,
    n_pairs = nrow(pairs),
    n_nonzero = n_entries,
    all_genes = all_genes,
    all_cells = all_cells,
    # `n_cells` is the number of columns in the matrix, whatever the cell space; this is how many
    # of them passed QC. Equal under the default -- an invariant worth being able to assert.
    n_cells_in_use = length(cells_in_use),
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
  # The settings as set_analysis_parameters() and run_qc() name them, which is what a low-MOI run
  # has to match. The two strings restate control_group_complement and run_permutations, which stay
  # for older readers. A slot an older sceptre object lacks is left out rather than written as
  # null, which the h5mu writer cannot store.
  cellwise <- slot_or_null(so, "cellwise_qc_thresholds")
  settings <- list(
    low_moi = so@low_moi,
    control_group = if (so@control_group_complement) "complement" else "nt_cells",
    resampling_mechanism = if (so@run_permutations) "permutations" else "crt",
    side = c("left", "both", "right")[so@side_code + 2L],
    multiple_testing_method = slot_or_null(so, "multiple_testing_method"),
    grna_assignment_method = slot_or_null(so, "grna_assignment_method"),
    response_n_umis_range = cellwise$response_n_umis_range,
    response_n_nonzero_range = cellwise$response_n_nonzero_range,
    p_mito_threshold = cellwise$p_mito_threshold,
    # Distinct cells carrying an NT gRNA; under nt_cells, the control group's size.
    n_nt_cells = length(unique(unlist(indiv_nt, use.names = FALSE)))
  )
  settings <- settings[!vapply(settings, function(v) is.null(v) || length(v) == 0, logical(1))]
  metadata <- c(metadata, settings)
  write_json(metadata, file.path(out_dir, "metadata.json"),
             auto_unbox = TRUE, pretty = TRUE)

  invisible(out_dir)
}
