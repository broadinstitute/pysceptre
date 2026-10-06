#!/usr/bin/env Rscript
# Ground truth for the low-MOI port, from the installed `sceptre` package.
#
# Writes one JSON fixture with three sections. Treat it like the other committed ground-truth JSONs
# in tests/validation/: if you change this script, DELETE lowmoi_ground_truth.json.gz and regenerate
# it, or the tests keep validating against the stale fixture.
#
#   samplers    exact draws of sceptre's two permutation samplers. Both reseed boost::mt19937(4) on
#               every call, so a draw sequence is a pure function of the arguments and the first k
#               draws of a long call equal a call with B = k.
#   unit        a small synthetic design pushed through the per-pair nt_cells path one internal
#               call at a time (combined-cell fits, pieces, run_low_level_test_full_v4).
#   end_to_end  sceptre's public pipeline on its own simulated example data, over
#               control_group in {nt_cells, complement} x resampling_mechanism in
#               {permutations, crt}. Each grid cell starts from the same import_data(moi = "low")
#               object: construct_positive_control_pairs, construct_trans_pairs(pairs_to_exclude =
#               "pc_pairs"), set_analysis_parameters(side = "both"), assign_grnas and run_qc with
#               their defaults, then run_calibration_check, run_power_check and
#               run_discovery_analysis (parallel = FALSE).
#
# Data: data(lowmoi_example_data, package = "sceptre") is simulated (sceptre's
# data-raw/DATASET_lowmoi_example_data.R), 100 genes x 1000 cells. The sceptredata package ships
# REAL data under the same name; never load it here, this fixture is committed.
#
# Provenance: the fixture records the sceptre version, its install SHA and the R version.
#
# Conventions
#   - Every index is 0-based and every index set is a JSON array, even at length 1.
#   - Cell indices in `end_to_end` are relative to `cells_in_use_0based` (the post-QC cells);
#     `cells_in_use_0based` itself is relative to the 1000 imported cells. In `unit` they index
#     the rows of `X`.
#   - Sampler draws keep R's order and are never sorted: with use_all_cells = FALSE only the first
#     n_trt entries of a draw are used. fisher_yates_samlper returns the last M slots of its
#     swap array, so draw[0] is the LAST element it selected.
#   - Tables are column-major objects {column: [values]}; NA is null.
#   - B = B1 + B2 + B3 draws per sampler call. Permutations: 499 + 4999 + 24999. CRT: B3 = 0.
#
# Layout
#   provenance: {sceptre_version, sceptre_remote_sha, r_version, unit_seed}
#   samplers:
#     fisher_yates[]:       {n_tot, M, B, draws_0based (B x M)}
#     hybrid_fisher_iwor[]: {N, m, M, B, draws_0based (B x M)}
#     pins[]:               {used_by, sampler, args, draw_index_0based[], draws_0based,
#                            sum_entries, sum_position_weighted}
#                           one per sampler call used below, at full B. sum_position_weighted
#                           is sum over draws of sum_j (j + 1) * draw[j].
#   unit:
#     n_cells, X (n_cells x 3, row-major), X_colnames, B1, B2, B3, side_code
#     genes[]:   {gene_id, y}
#     targets[]: {target_id, cells_0based, n_trt, combined_cells_0based,
#                 logistic_fitted_probabilities, logistic_coefs}
#                combined_cells_0based = c(target cells, all_nt_idxs), the nt_cells cell order.
#     nt_grnas[]: {grna_id, cells_0based, positions_in_nt_pool_0based}
#     all_nt_idxs_0based: the NT pool in R's order (gRNA concatenation, not sorted)
#     permutation_sampler: {sampler, N, m, M, B}, shared by every pair
#     pairs[]: {target_id, gene_id, n_trt, fitted_coefs, theta, mu, w, a,
#               perm, perm_no_approximation, crt}
#               mu, w, a are over combined_cells_0based. Each result is
#               {p, z_orig, fc, se_over_root_n, stage, sn_params, B1, B2, B3}.
#               perm: skew_normal on the shared permutation draws (use_all_cells = FALSE).
#               perm_no_approximation: same draws, fit_parametric_curve = FALSE, B2 = 0, so a
#                 pair with stage-1 p <= 0.02 reads draws [B1, B1 + B3) at stage 3.
#               crt: crt_index_sampler_fast on the target's logistic fit (use_all_cells = TRUE).
#     calibration: the nt_cells calibration path on the same design.
#       nt_pool_cells_0based, calibration_group_size, permutation_sampler {sampler, n_tot, M, B}
#       genes[]:  {gene_id, fitted_coefs, theta, mu, w, a}, outer regression over the NT pool
#       groups[]: {grna_group, trt_positions_0based, n_trt, logistic_fitted_probabilities,
#                  logistic_coefs}; positions index the NT pool
#       pairs[]:  {gene_id, grna_group, n_trt, perm, crt}
#   end_to_end:
#     dataset, formula, n_cells_imported, cells_in_use_0based, cell_removal_metrics,
#     response_ids, response_matrix (genes x cells_in_use, integer counts),
#     covariate_colnames, covariate_matrix (cells_in_use x p, row-major)
#     grna_assignments: {targets: {target: cells}, nt_grnas: {grna_id: cells},
#                        all_nt_idxs_0based, nt_positions_in_pool_0based: {grna_id: positions}}
#       Identical in all four grid cells (asserted). nt_grnas is in cell space under both control
#       groups; all_nt_idxs_0based and the positions are R's own nt_cells representation.
#     grid[]: {control_group, resampling_mechanism, resampling_approximation,
#              grna_integration_strategy, B1, B2, B3, side_code, multiple_testing_method,
#              multiple_testing_alpha, n_nonzero_trt_thresh, n_nonzero_cntrl_thresh,
#              calibration_group_size, n_calibration_pairs, n_ok_discovery_pairs,
#              n_ok_positive_control_pairs,
#              permutation_samplers: {calibration, power_check, discovery} or null for crt,
#              discovery_pairs_with_info, positive_control_pairs_with_info, negative_control_pairs,
#              calibration_result, power_result, discovery_result}
#       Results come from output_amount = 2, keeping response_id, grna_target, n_nonzero_trt,
#       n_nonzero_cntrl, pass_qc, p_value, log_2_fold_change, significant (where present),
#       stage and z_orig. Rows keep R's order, sorted by p_value then response_id, so match
#       rows on (response_id, grna_target).
#
# Usage:
#   Rscript scripts/dump_lowmoi_ground_truth.R tests/validation/lowmoi_ground_truth.json
# The committed fixture is that JSON gzipped, tests/validation/lowmoi_ground_truth.json.gz;
# conftest.py regenerates and gzips it when the .gz is missing.

suppressPackageStartupMessages({
  library(sceptre)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
out_path <- if (length(args) >= 1) args[[1]] else "tests/validation/lowmoi_ground_truth.json"

ns <- asNamespace("sceptre")
sc <- function(f) get(f, envir = ns)

arr <- function(x) I(x)
idx0 <- function(v) I(as.integer(v) - 1L)

B1 <- 499L
B2 <- 4999L
B3_perm <- 24999L
B3_crt <- 0L
B_perm <- B1 + B2 + B3_perm
side_code <- 0L
draw_pin_idxs <- c(0L, 1L, 2L, B1, B1 + B2, B_perm - 1L)

sceptre_remote_sha <- utils::packageDescription("sceptre")$RemoteSha
provenance <- list(
  sceptre_version = as.character(utils::packageVersion("sceptre")),
  sceptre_remote_sha = if (is.null(sceptre_remote_sha)) NA_character_ else sceptre_remote_sha,
  r_version = R.version.string
)
cat(sprintf("sceptre %s (%s), %s\n", provenance$sceptre_version, provenance$sceptre_remote_sha, provenance$r_version))

wall <- system.time({

# ---- A. samplers ------------------------------------------------------------------------------

draws_list <- function(ptr) lapply(sc("synth_idx_list_to_r_list")(ptr), as.integer)
draws_matrix <- function(ptr) do.call(rbind, draws_list(ptr))

make_sampler <- function(args) {
  if (args$sampler == "fisher_yates") {
    sc("fisher_yates_samlper")(n_tot = args$n_tot, M = args$M, B = args$B)
  } else {
    sc("hybrid_fisher_iwor_sampler")(N = args$N, m = args$m, M = args$M, B = args$B)
  }
}

fy_cases <- list(c(10L, 3L, 5L), c(40L, 7L, 20L), c(6L, 6L, 4L), c(100L, 1L, 10L), c(30L, 10L, 50L))
fisher_yates <- lapply(fy_cases, function(cs) {
  list(n_tot = cs[1], M = cs[2], B = cs[3],
       draws_0based = draws_matrix(sc("fisher_yates_samlper")(n_tot = cs[1], M = cs[2], B = cs[3])))
})

iwor_cases <- list(c(10L, 2L, 5L, 5L), c(30L, 3L, 9L, 20L), c(25L, 4L, 4L, 10L), c(8L, 1L, 6L, 10L),
                   c(50L, 5L, 30L, 8L))
hybrid_fisher_iwor <- lapply(iwor_cases, function(cs) {
  list(N = cs[1], m = cs[2], M = cs[3], B = cs[4],
       draws_0based = draws_matrix(sc("hybrid_fisher_iwor_sampler")(N = cs[1], m = cs[2], M = cs[3], B = cs[4])))
})

pin_sampler <- function(used_by, args) {
  d <- draws_list(make_sampler(args))
  stopifnot(length(d) == args$B)
  list(
    used_by = used_by,
    sampler = args$sampler,
    args = args[setdiff(names(args), "sampler")],
    draw_index_0based = arr(draw_pin_idxs[draw_pin_idxs < args$B]),
    draws_0based = do.call(rbind, d[draw_pin_idxs[draw_pin_idxs < args$B] + 1L]),
    sum_entries = sum(vapply(d, function(v) sum(as.numeric(v)), numeric(1))),
    sum_position_weighted = sum(vapply(d, function(v) sum(as.numeric(v) * seq_along(v)), numeric(1)))
  )
}

# ---- B. unit ----------------------------------------------------------------------------------

unit_seed <- 4L
set.seed(unit_seed)
n_cells <- 400L
X <- cbind(intercept = rep(1, n_cells), cov1 = rnorm(n_cells), cov2 = rnorm(n_cells))

# Every cell carries one gRNA, as after low-MOI QC; "other" cells sit on untested targets.
target_sizes <- c(target_1 = 20L, target_2 = 40L, target_3 = 60L)
nt_sizes <- c(nt_1 = 30L, nt_2 = 35L, nt_3 = 40L, nt_4 = 45L)
n_other <- n_cells - sum(target_sizes) - sum(nt_sizes)
labels <- sample(rep(c(names(target_sizes), names(nt_sizes), "other"), c(target_sizes, nt_sizes, n_other)))
trt <- lapply(stats::setNames(names(target_sizes), names(target_sizes)), function(t) which(labels == t))
indiv_nt <- lapply(stats::setNames(names(nt_sizes), names(nt_sizes)), function(g) which(labels == g))
nt_l <- sc("update_indiv_grna_assignments_for_nt_cells")(indiv_nt)
all_nt <- nt_l$all_nt_idxs
nt_pos <- nt_l$indiv_nt_grna_idxs
N_nt <- length(all_nt)

gene_specs <- list(
  gene_1 = list(beta = c(1.0, 0.3, -0.2), theta = 5, effect = c()),
  gene_2 = list(beta = c(0.7, -0.25, 0.15), theta = 1.5, effect = c(target_2 = 0.3)),
  gene_3 = list(beta = c(1.5, 0.1, 0.2), theta = 30, effect = c(target_1 = 2.5, target_3 = 0.6)),
  gene_4 = list(beta = c(-2.0, 0.2, 0.1), theta = 0.3, effect = c(target_3 = 6)),
  # moderate: its stage-3 p sits above the 2 / (1 + B3) floor, so the draw count matters
  gene_5 = list(beta = c(1.0, -0.2, 0.25), theta = 5, effect = c(target_2 = 0.9))
)
y_by_gene <- lapply(gene_specs, function(spec) {
  mu <- exp(as.numeric(X %*% spec$beta))
  for (t in names(spec$effect)) mu[trt[[t]]] <- mu[trt[[t]]] * spec$effect[[t]]
  as.integer(stats::rnbinom(n_cells, mu = mu, size = spec$theta))
})

result_fields <- function(r, b1, b2, b3) {
  list(p = r$p, z_orig = r$z_orig, fc = r$fc, se_over_root_n = r[["se_over_root_n"]],
       stage = as.integer(r$stage), sn_params = arr(as.numeric(r$sn_params)),
       B1 = b1, B2 = b2, B3 = b3)
}

unit_perm_args <- list(sampler = "hybrid_fisher_iwor", N = N_nt, m = min(target_sizes),
                       M = max(target_sizes), B = B_perm)
unit_perm_ptr <- make_sampler(unit_perm_args)

unit_targets <- list()
unit_crt_ptrs <- list()
for (t in names(trt)) {
  comb <- c(trt[[t]], all_nt)
  n_trt <- length(trt[[t]])
  fitted_probabilities <- sc("perform_grna_precomputation")(
    trt_idxs = seq_len(n_trt), covariate_matrix = X[comb, ], return_fitted_values = TRUE
  )
  logistic_coefs <- sc("perform_grna_precomputation")(
    trt_idxs = seq_len(n_trt), covariate_matrix = X[comb, ], return_fitted_values = FALSE
  )
  unit_crt_ptrs[[t]] <- sc("crt_index_sampler_fast")(fitted_probabilities = fitted_probabilities, B = B1 + B2 + B3_crt)
  unit_targets[[t]] <- list(
    target_id = t, cells_0based = idx0(trt[[t]]), n_trt = n_trt,
    combined_cells_0based = idx0(comb),
    logistic_fitted_probabilities = arr(as.numeric(fitted_probabilities)),
    logistic_coefs = arr(as.numeric(logistic_coefs))
  )
}

run_test <- function(y, pieces, trt_idxs, n_trt, use_all_cells, ptr, b1, b2, b3, fit_curve) {
  sc("run_low_level_test_full_v4")(
    y = y, mu = pieces$mu, a = pieces$a, w = pieces$w, D = pieces$D,
    trt_idxs = trt_idxs, n_trt = n_trt, use_all_cells = use_all_cells, synthetic_idxs = ptr,
    B1 = b1, B2 = b2, B3 = b3, fit_parametric_curve = fit_curve,
    return_resampling_dist = FALSE, side_code = side_code
  )
}

unit_pairs <- list()
for (t in names(trt)) {
  for (g in names(gene_specs)) {
    comb <- c(trt[[t]], all_nt)
    n_trt <- length(trt[[t]])
    y_c <- y_by_gene[[g]][comb]
    X_c <- X[comb, ]
    rp <- sc("perform_response_precomputation")(expressions = y_c, covariate_matrix = X_c)
    pieces <- sc("compute_precomputation_pieces")(
      expression_vector = y_c, covariate_matrix = X_c,
      fitted_coefs = rp$fitted_coefs, theta = rp$theta, full_test_stat = TRUE
    )
    r_perm <- run_test(y_c, pieces, seq_len(n_trt), n_trt, FALSE, unit_perm_ptr, B1, B2, B3_perm, TRUE)
    r_na <- run_test(y_c, pieces, seq_len(n_trt), n_trt, FALSE, unit_perm_ptr, B1, 0L, B3_perm, FALSE)
    r_crt <- run_test(y_c, pieces, seq_len(n_trt), n_trt, TRUE, unit_crt_ptrs[[t]], B1, B2, B3_crt, TRUE)
    unit_pairs[[length(unit_pairs) + 1L]] <- list(
      target_id = t, gene_id = g, n_trt = n_trt,
      fitted_coefs = arr(as.numeric(rp$fitted_coefs)), theta = rp$theta,
      mu = arr(as.numeric(pieces$mu)), w = arr(as.numeric(pieces$w)), a = arr(as.numeric(pieces$a)),
      perm = result_fields(r_perm, B1, B2, B3_perm),
      perm_no_approximation = result_fields(r_na, B1, 0L, B3_perm),
      crt = result_fields(r_crt, B1, B2, B3_crt)
    )
  }
}
unit_stages <- vapply(unit_pairs, function(u) u$perm$stage, integer(1))
unit_na_stages <- vapply(unit_pairs, function(u) u$perm_no_approximation$stage, integer(1))
unit_na_p <- vapply(unit_pairs, function(u) u$perm_no_approximation$p, numeric(1))
cat("unit perm stages:", unit_stages, "\n")
cat("unit perm_no_approximation stages:", unit_na_stages, "\n")
stopifnot(any(unit_stages == 1L), any(unit_stages >= 2L),
          any(unit_na_stages == 3L & unit_na_p > 2 / (1 + B3_perm)))

# nt_cells calibration: the whole universe is the NT pool, positions index it.
calibration_group_size_unit <- 2L
X_nt <- X[all_nt, ]
cal_sizes <- sort(vapply(nt_pos, length, integer(1)), decreasing = TRUE)
unit_cal_args <- list(sampler = "fisher_yates", n_tot = N_nt,
                      M = sum(cal_sizes[seq_len(calibration_group_size_unit)]), B = B_perm)
unit_cal_ptr <- make_sampler(unit_cal_args)
cal_group_names <- c("nt_1&nt_2", "nt_3&nt_4")

cal_genes <- list()
cal_pieces <- list()
for (g in names(gene_specs)) {
  y_nt <- y_by_gene[[g]][all_nt]
  rp <- sc("perform_response_precomputation")(expressions = y_nt, covariate_matrix = X_nt)
  pieces <- sc("compute_precomputation_pieces")(
    expression_vector = y_nt, covariate_matrix = X_nt,
    fitted_coefs = rp$fitted_coefs, theta = rp$theta, full_test_stat = TRUE
  )
  cal_pieces[[g]] <- pieces
  cal_genes[[g]] <- list(
    gene_id = g, fitted_coefs = arr(as.numeric(rp$fitted_coefs)), theta = rp$theta,
    mu = arr(as.numeric(pieces$mu)), w = arr(as.numeric(pieces$w)), a = arr(as.numeric(pieces$a))
  )
}
cal_trt_pos <- lapply(stats::setNames(cal_group_names, cal_group_names), function(grp) {
  as.integer(unlist(nt_pos[strsplit(grp, split = "&", fixed = TRUE)[[1]]], use.names = FALSE))
})
cal_groups <- list()
cal_crt_ptrs <- list()
for (grp in cal_group_names) {
  trt_pos <- cal_trt_pos[[grp]]
  fitted_probabilities <- sc("perform_grna_precomputation")(
    trt_idxs = trt_pos, covariate_matrix = X_nt, return_fitted_values = TRUE
  )
  logistic_coefs <- sc("perform_grna_precomputation")(
    trt_idxs = trt_pos, covariate_matrix = X_nt, return_fitted_values = FALSE
  )
  cal_crt_ptrs[[grp]] <- sc("crt_index_sampler_fast")(fitted_probabilities = fitted_probabilities, B = B1 + B2 + B3_crt)
  cal_groups[[grp]] <- list(
    grna_group = grp, trt_positions_0based = idx0(trt_pos), n_trt = length(trt_pos),
    logistic_fitted_probabilities = arr(as.numeric(fitted_probabilities)),
    logistic_coefs = arr(as.numeric(logistic_coefs))
  )
}
cal_pairs <- list()
for (g in names(gene_specs)) {
  for (grp in cal_group_names) {
    trt_pos <- cal_trt_pos[[grp]]
    y_nt <- y_by_gene[[g]][all_nt]
    r_perm <- run_test(y_nt, cal_pieces[[g]], trt_pos, length(trt_pos), FALSE, unit_cal_ptr, B1, B2, B3_perm, TRUE)
    r_crt <- run_test(y_nt, cal_pieces[[g]], trt_pos, length(trt_pos), TRUE, cal_crt_ptrs[[grp]], B1, B2, B3_crt, TRUE)
    cal_pairs[[length(cal_pairs) + 1L]] <- list(
      gene_id = g, grna_group = grp, n_trt = length(trt_pos),
      perm = result_fields(r_perm, B1, B2, B3_perm),
      crt = result_fields(r_crt, B1, B2, B3_crt)
    )
  }
}

unit <- list(
  n_cells = n_cells,
  X = unname(X),
  X_colnames = arr(colnames(X)),
  B1 = B1, B2 = B2, B3 = B3_perm, side_code = side_code,
  genes = unname(lapply(names(y_by_gene), function(g) list(gene_id = g, y = arr(y_by_gene[[g]])))),
  targets = unname(unit_targets),
  nt_grnas = unname(lapply(names(indiv_nt), function(g) {
    list(grna_id = g, cells_0based = idx0(indiv_nt[[g]]), positions_in_nt_pool_0based = idx0(nt_pos[[g]]))
  })),
  all_nt_idxs_0based = idx0(all_nt),
  permutation_sampler = unit_perm_args,
  pairs = unit_pairs,
  calibration = list(
    nt_pool_cells_0based = idx0(all_nt),
    calibration_group_size = calibration_group_size_unit,
    permutation_sampler = unit_cal_args,
    genes = unname(cal_genes),
    groups = unname(cal_groups),
    pairs = cal_pairs
  )
)

# ---- C. end_to_end ----------------------------------------------------------------------------

data(lowmoi_example_data, package = "sceptre", envir = environment())
d <- lowmoi_example_data
obj_imported <- import_data(
  response_matrix = d$response_matrix, grna_matrix = d$grna_matrix,
  grna_target_data_frame = d$grna_target_data_frame, moi = "low",
  extra_covariates = d$extra_covariates
)
positive_control_pairs <- construct_positive_control_pairs(obj_imported)
discovery_pairs <- construct_trans_pairs(
  sceptre_object = obj_imported, positive_control_pairs = positive_control_pairs,
  pairs_to_exclude = "pc_pairs"
)

perm_sampler_args <- function(obj, calibration_check) {
  ga <- obj@grna_assignments
  B <- obj@B1 + obj@B2 + obj@B3
  if (calibration_check) {
    sizes <- sort(vapply(ga$indiv_nt_grna_idxs, length, integer(1)), decreasing = TRUE)
    M <- sum(sizes[seq_len(obj@calibration_group_size)])
    n_tot <- if (obj@control_group_complement) length(obj@cells_in_use) else length(ga$all_nt_idxs)
    list(sampler = "fisher_yates", n_tot = n_tot, M = M, B = B)
  } else if (!obj@control_group_complement) {
    sizes <- vapply(ga$grna_group_idxs, length, integer(1))
    sizes <- sizes[sizes != 0L]
    list(sampler = "hybrid_fisher_iwor", N = length(ga$all_nt_idxs), m = min(sizes), M = max(sizes), B = B)
  } else {
    list(sampler = "fisher_yates", n_tot = length(obj@cells_in_use),
         M = max(vapply(ga$grna_group_idxs, length, integer(1))), B = B)
  }
}

# The recorded arguments must reproduce the draws R's own pipeline helper makes.
check_sampler_args <- function(obj, calibration_check, args) {
  ga <- obj@grna_assignments
  theirs <- draws_list(sc("get_synthetic_permutation_idxs")(
    grna_assignments = ga, B = args$B, calibration_check = calibration_check,
    control_group_complement = obj@control_group_complement,
    calibration_group_size = obj@calibration_group_size, n_cells = length(obj@cells_in_use)
  ))
  ours <- draws_list(make_sampler(args))
  stopifnot(identical(theirs, ours))
}

table_columns <- function(df, cols) {
  df <- as.data.frame(df)
  cols <- intersect(cols, colnames(df))
  out <- lapply(cols, function(cl) {
    v <- df[[cl]]
    if (is.factor(v)) v <- as.character(v)
    arr(v)
  })
  stats::setNames(out, cols)
}
info_cols <- c("response_id", "grna_group", "n_nonzero_trt", "n_nonzero_cntrl", "pass_qc")
result_cols <- c("response_id", "grna_target", "n_nonzero_trt", "n_nonzero_cntrl", "pass_qc",
                 "p_value", "log_2_fold_change", "significant", "stage", "z_orig")

grid_spec <- list(
  c("nt_cells", "permutations"), c("nt_cells", "crt"),
  c("complement", "permutations"), c("complement", "crt")
)
objs <- list()
grid <- list()
for (gs in grid_spec) {
  control_group <- gs[1]
  resampling_mechanism <- gs[2]
  label <- paste(control_group, resampling_mechanism, sep = "/")
  cat("running", label, "\n")
  obj <- obj_imported |>
    set_analysis_parameters(
      discovery_pairs = discovery_pairs, positive_control_pairs = positive_control_pairs,
      side = "both", control_group = control_group, resampling_mechanism = resampling_mechanism
    ) |>
    assign_grnas(print_progress = FALSE) |>
    run_qc() |>
    run_calibration_check(parallel = FALSE, print_progress = FALSE, output_amount = 2) |>
    run_power_check(parallel = FALSE, print_progress = FALSE, output_amount = 2) |>
    run_discovery_analysis(parallel = FALSE, print_progress = FALSE, output_amount = 2)
  objs[[label]] <- obj

  samplers <- NA
  if (obj@run_permutations) {
    samplers <- list(
      calibration = perm_sampler_args(obj, TRUE),
      power_check = perm_sampler_args(obj, FALSE),
      discovery = perm_sampler_args(obj, FALSE)
    )
    check_sampler_args(obj, TRUE, samplers$calibration)
    check_sampler_args(obj, FALSE, samplers$discovery)
  }

  grid[[length(grid) + 1L]] <- list(
    control_group = control_group,
    resampling_mechanism = resampling_mechanism,
    resampling_approximation = obj@resampling_approximation,
    grna_integration_strategy = obj@grna_integration_strategy,
    B1 = obj@B1, B2 = obj@B2, B3 = obj@B3,
    side_code = obj@side_code,
    multiple_testing_method = obj@multiple_testing_method,
    multiple_testing_alpha = obj@multiple_testing_alpha,
    n_nonzero_trt_thresh = obj@n_nonzero_trt_thresh,
    n_nonzero_cntrl_thresh = obj@n_nonzero_cntrl_thresh,
    calibration_group_size = obj@calibration_group_size,
    n_calibration_pairs = obj@n_calibration_pairs,
    n_ok_discovery_pairs = obj@n_ok_discovery_pairs,
    n_ok_positive_control_pairs = obj@n_ok_positive_control_pairs,
    permutation_samplers = samplers,
    discovery_pairs_with_info = table_columns(obj@discovery_pairs_with_info, info_cols),
    positive_control_pairs_with_info = table_columns(obj@positive_control_pairs_with_info, info_cols),
    negative_control_pairs = table_columns(obj@negative_control_pairs, info_cols),
    calibration_result = table_columns(obj@calibration_result, result_cols),
    power_result = table_columns(obj@power_result, result_cols),
    discovery_result = table_columns(obj@discovery_result, result_cols)
  )
}

# gRNA assignments in cell space, relative to cells_in_use.
cell_space_nt <- function(obj) {
  ga <- obj@grna_assignments
  if (obj@control_group_complement) {
    lapply(ga$indiv_nt_grna_idxs, as.integer)
  } else {
    lapply(ga$indiv_nt_grna_idxs, function(pos) as.integer(ga$all_nt_idxs[pos]))
  }
}
ref <- objs[["nt_cells/permutations"]]
for (label in names(objs)) {
  o <- objs[[label]]
  stopifnot(
    identical(o@cells_in_use, ref@cells_in_use),
    identical(lapply(o@grna_assignments$grna_group_idxs, as.integer),
              lapply(ref@grna_assignments$grna_group_idxs, as.integer)),
    identical(cell_space_nt(o), cell_space_nt(ref))
  )
}
plain <- function(df) as.list(as.data.frame(df))
for (cg in c("nt_cells", "complement")) {
  a <- objs[[paste0(cg, "/permutations")]]
  b <- objs[[paste0(cg, "/crt")]]
  for (s in c("discovery_pairs_with_info", "positive_control_pairs_with_info", "negative_control_pairs")) {
    stopifnot(identical(plain(methods::slot(a, s)), plain(methods::slot(b, s))))
  }
}

cells_in_use <- ref@cells_in_use
ga_ref <- ref@grna_assignments
resp <- as.matrix(get_response_matrix(ref))[, cells_in_use, drop = FALSE]
stopifnot(all(resp == round(resp)))
storage.mode(resp) <- "integer"
covariates <- ref@covariate_matrix[cells_in_use, , drop = FALSE]
nt_cells_space <- cell_space_nt(ref)
stopifnot(identical(sort(unlist(nt_cells_space, use.names = FALSE)), sort(as.integer(ga_ref$all_nt_idxs))))

end_to_end <- list(
  dataset = "data(lowmoi_example_data, package = \"sceptre\"), simulated",
  formula = paste(deparse(ref@formula_object), collapse = " "),
  n_cells_imported = ncol(get_response_matrix(ref)),
  cells_in_use_0based = idx0(cells_in_use),
  cell_removal_metrics = as.list(ref@cell_removal_metrics),
  response_ids = arr(rownames(resp)),
  response_matrix = unname(resp),
  covariate_colnames = arr(colnames(covariates)),
  covariate_matrix = unname(covariates),
  grna_assignments = list(
    targets = lapply(ga_ref$grna_group_idxs, idx0),
    nt_grnas = lapply(nt_cells_space, idx0),
    all_nt_idxs_0based = idx0(ga_ref$all_nt_idxs),
    nt_positions_in_pool_0based = lapply(ga_ref$indiv_nt_grna_idxs, idx0)
  ),
  grid = grid
)

# ---- pins: every sampler call made above, at full B --------------------------------------------

pin_calls <- list(
  list(used_by = "unit/perm", args = unit_perm_args),
  list(used_by = "unit/calibration/perm", args = unit_cal_args)
)
for (cell in grid) {
  if (!identical(cell$permutation_samplers, NA)) {
    for (an in names(cell$permutation_samplers)) {
      pin_calls[[length(pin_calls) + 1L]] <- list(
        used_by = paste0("end_to_end/", cell$control_group, "/", an),
        args = cell$permutation_samplers[[an]]
      )
    }
  }
}
pin_keys <- vapply(pin_calls, function(p) paste(unlist(p$args), collapse = ","), character(1))
pins <- lapply(split(pin_calls, factor(pin_keys, levels = unique(pin_keys))), function(group) {
  pin_sampler(paste(vapply(group, function(p) p$used_by, character(1)), collapse = "; "), group[[1]]$args)
})

out <- list(
  provenance = c(provenance, list(unit_seed = unit_seed)),
  samplers = list(fisher_yates = fisher_yates, hybrid_fisher_iwor = hybrid_fisher_iwor, pins = unname(pins)),
  unit = unit,
  end_to_end = end_to_end
)
write(toJSON(out, auto_unbox = TRUE, digits = 15, na = "null", dataframe = "columns"), out_path)

})

summarise_table <- function(tab) {
  n <- length(tab[[1]])
  pass <- if (!is.null(tab$pass_qc)) sum(unclass(tab$pass_qc), na.rm = TRUE) else NA
  sig <- if (!is.null(tab$significant)) sum(unclass(tab$significant), na.rm = TRUE) else NA
  st <- table(unclass(tab$stage))
  stg <- if (length(st) > 0L) paste(names(st), st, sep = ":", collapse = " ") else ""
  sprintf("%5d rows, %5s pass_qc, %4s significant, stages %s", n, pass, sig, stg)
}
for (cell in grid) {
  cat(sprintf("%s/%s\n", cell$control_group, cell$resampling_mechanism))
  for (tn in c("discovery_pairs_with_info", "positive_control_pairs_with_info", "negative_control_pairs",
               "calibration_result", "power_result", "discovery_result")) {
    cat(sprintf("  %-34s %s\n", tn, summarise_table(cell[[tn]])))
  }
}
cat(sprintf("wrote %s (%.0f bytes) in %.1f s\n", out_path, file.size(out_path), wall[["elapsed"]]))
