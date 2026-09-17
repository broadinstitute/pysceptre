#!/usr/bin/env Rscript
# Dumps ground-truth intermediates from the real, installed `sceptre` package
# by calling its internal (`:::`) functions directly on small hand-built
# matrices, bypassing the sceptre_object/QC/assignment machinery entirely.
# Output: a single JSON file consumed by pysceptre's validation test suite.
#
# Usage: Rscript dump_r_ground_truth.R <output.json> [seed]

suppressWarnings(suppressMessages({
  library(sceptre)
  library(jsonlite)
}))

args <- commandArgs(trailingOnly = TRUE)
out_path <- if (length(args) >= 1) args[[1]] else "ground_truth.json"
seed <- if (length(args) >= 2) as.integer(args[[2]]) else 4L
set.seed(seed)

ns <- asNamespace("sceptre")
sc <- function(f) get(f, envir = ns)

n_cells <- 400L
p_covariates <- 3L # intercept + 2 continuous covariates

# ---- design matrix (intercept + 2 continuous covariates), matches what
# auto_construct_formula_object would produce for simple numeric covariates
X <- cbind(
  intercept = rep(1, n_cells),
  cov1 = rnorm(n_cells),
  cov2 = rnorm(n_cells)
)

# ---- a handful of "genes": Poisson/NB-like count vectors regressed on X
n_genes <- 4L
true_theta <- c(5, 20, 0.8, 200)
gene_results <- list()

for (g in seq_len(n_genes)) {
  beta <- c(1.2, 0.3, -0.2) + rnorm(3, sd = 0.05)
  mu_true <- exp(X %*% beta)
  y <- rnbinom(n_cells, mu = mu_true, size = true_theta[g])

  response_precomp <- sc("perform_response_precomputation")(expressions = y, covariate_matrix = X)
  pieces <- sc("compute_precomputation_pieces")(
    expression_vector = y, covariate_matrix = X,
    fitted_coefs = response_precomp$fitted_coefs, theta = response_precomp$theta,
    full_test_stat = TRUE
  )

  gene_results[[g]] <- list(
    gene_id = paste0("gene_", g),
    y = y,
    fitted_coefs = as.numeric(response_precomp$fitted_coefs),
    theta = response_precomp$theta,
    mu = as.numeric(pieces$mu),
    w = as.numeric(pieces$w),
    a = as.numeric(pieces$a),
    D = pieces$D # p x n matrix
  )
}

# ---- a handful of "targets": treatment index sets + logistic precomputation + CRT draws
n_targets <- 3L
target_results <- list()
B1 <- 499L
B2 <- 4999L
B3 <- 0L
B_total <- B1 + B2 + B3

for (t in seq_len(n_targets)) {
  n_trt <- sample(20:80, 1)
  trt_idxs <- sort(sample(seq_len(n_cells), n_trt)) # 1-based, R convention

  fitted_probabilities <- sc("perform_grna_precomputation")(
    trt_idxs = trt_idxs, covariate_matrix = X, return_fitted_values = TRUE
  )
  logistic_coefs <- sc("perform_grna_precomputation")(
    trt_idxs = trt_idxs, covariate_matrix = X, return_fitted_values = FALSE
  )

  synthetic_idxs_ptr <- sc("crt_index_sampler_fast")(fitted_probabilities = fitted_probabilities, B = B_total)
  synthetic_idxs_r <- sc("synth_idx_list_to_r_list")(synthetic_idxs_ptr)
  # convert to 0-based for downstream Python consumption
  synthetic_idxs_0based <- lapply(synthetic_idxs_r, function(v) as.integer(v) - 1L)

  target_results[[t]] <- list(
    target_id = paste0("target_", t),
    trt_idxs_1based = as.integer(trt_idxs),
    n_trt = n_trt,
    fitted_probabilities = as.numeric(fitted_probabilities),
    logistic_coefs = as.numeric(logistic_coefs),
    synthetic_idxs_0based = synthetic_idxs_0based
  )
}

# ---- full per-pair test: every gene x every target, via run_low_level_test_full_v4
side_code <- 0L # "both"
pair_results <- list()
idx <- 1L
for (g in seq_len(n_genes)) {
  for (t in seq_len(n_targets)) {
    gene <- gene_results[[g]]
    target <- target_results[[t]]

    synthetic_idxs_ptr <- sc("crt_index_sampler_fast")(
      fitted_probabilities = target$fitted_probabilities, B = B_total
    )

    result <- sc("run_low_level_test_full_v4")(
      y = gene$y,
      mu = gene$mu,
      a = gene$a,
      w = gene$w,
      D = matrix(unlist(gene$D), nrow = nrow(gene_results[[g]]$D)),
      trt_idxs = target$trt_idxs_1based,
      n_trt = target$n_trt,
      use_all_cells = TRUE,
      synthetic_idxs = synthetic_idxs_ptr,
      B1 = B1, B2 = B2, B3 = B3,
      fit_parametric_curve = TRUE,
      return_resampling_dist = TRUE,
      side_code = side_code
    )

    pair_results[[idx]] <- list(
      gene_id = gene$gene_id,
      target_id = target$target_id,
      p_value = result$p,
      z_orig = result$z_orig,
      fold_change = result$fc,
      se_fold_change = result$se,
      stage = result$stage,
      sn_params = as.numeric(result$sn_params),
      resampling_dist = as.numeric(result$resampling_dist)
    )
    idx <- idx + 1L
  }
}

# ---- one genuine-effect pair, to exercise stage 2+ (skew-normal) escalation
# end-to-end (all null-signal pairs above only ever hit stage 1)
signal_target <- target_results[[1]]
signal_y <- gene_results[[1]]$y
signal_y[signal_target$trt_idxs_1based] <- signal_y[signal_target$trt_idxs_1based] * 6L + 5L
signal_response_precomp <- sc("perform_response_precomputation")(expressions = signal_y, covariate_matrix = X)
signal_pieces <- sc("compute_precomputation_pieces")(
  expression_vector = signal_y, covariate_matrix = X,
  fitted_coefs = signal_response_precomp$fitted_coefs, theta = signal_response_precomp$theta,
  full_test_stat = TRUE
)
signal_synthetic_idxs_ptr <- sc("crt_index_sampler_fast")(
  fitted_probabilities = signal_target$fitted_probabilities, B = B_total
)
signal_result <- sc("run_low_level_test_full_v4")(
  y = signal_y,
  mu = signal_pieces$mu,
  a = signal_pieces$a,
  w = signal_pieces$w,
  D = signal_pieces$D,
  trt_idxs = signal_target$trt_idxs_1based,
  n_trt = signal_target$n_trt,
  use_all_cells = TRUE,
  synthetic_idxs = signal_synthetic_idxs_ptr,
  B1 = B1, B2 = B2, B3 = B3,
  fit_parametric_curve = TRUE,
  return_resampling_dist = TRUE,
  side_code = side_code
)
signal_pair <- list(
  gene_id = "signal_gene",
  target_id = signal_target$target_id,
  y = signal_y,
  fitted_coefs = as.numeric(signal_response_precomp$fitted_coefs),
  theta = signal_response_precomp$theta,
  mu = as.numeric(signal_pieces$mu),
  w = as.numeric(signal_pieces$w),
  a = as.numeric(signal_pieces$a),
  D = signal_pieces$D,
  p_value = signal_result$p,
  z_orig = signal_result$z_orig,
  fold_change = signal_result$fc,
  se_fold_change = signal_result$se,
  stage = signal_result$stage,
  sn_params = as.numeric(signal_result$sn_params),
  resampling_dist = as.numeric(signal_result$resampling_dist)
)

# ---- standalone unit-level ground truth for isolated formula validation
set.seed(seed + 1L)
null_stats_for_sn <- rnorm(4999, mean = 0.1, sd = 1.05) + rgamma(4999, shape = 2, rate = 4) - 0.5
sn_fit <- sc("fit_skew_normal_funct")(null_stats_for_sn)
z_test <- 1.8
sn_eval <- sc("fit_and_evaluate_skew_normal")(z_test, null_stats_for_sn, side_code)
emp_p <- sc("compute_empirical_p_value")(null_stats_for_sn, z_test, side_code)

y_theta_test <- rnbinom(500, mu = 8, size = 3.5)
mu_theta_test <- rep(mean(y_theta_test), 500)
theta_est <- sc("estimate_theta")(
  y = y_theta_test, mu = mu_theta_test, dfr = 499,
  limit = 50, eps = (.Machine$double.eps)^(1 / 4)
)

standalone <- list(
  skew_normal = list(
    input_null_stats = as.numeric(null_stats_for_sn),
    fit = as.numeric(sn_fit),
    z_test = z_test,
    eval_result = as.numeric(sn_eval)
  ),
  empirical_p = list(
    input_null_stats = as.numeric(null_stats_for_sn),
    z_test = z_test,
    side_code = side_code,
    p_value = emp_p
  ),
  nb_theta = list(
    y = as.numeric(y_theta_test),
    mu = as.numeric(mu_theta_test),
    dfr = 499,
    theta_result = as.numeric(theta_est)
  )
)

out <- list(
  seed = seed,
  n_cells = n_cells,
  X = X,
  B1 = B1, B2 = B2, B3 = B3,
  genes = gene_results,
  targets = target_results,
  pairs = pair_results,
  signal_pair = signal_pair,
  standalone = standalone
)

write(toJSON(out, auto_unbox = TRUE, digits = 15), out_path)
cat("Wrote ground truth to", out_path, "\n")
