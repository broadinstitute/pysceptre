#!/usr/bin/env Rscript
# Ground truth for the port of sceptre's mixture assignment (pysceptre.assignment.mixture).
#
#   Rscript scripts/dump_mixture_ground_truth.R <out.json>
#
# Writes one JSON fixture; tests/validation/conftest.py gzips it to mixture_ground_truth.json.gz
# (mtime 0). If you change this script, DELETE the .gz and regenerate it; the fixture records
# this script's md5 and a test compares it with the script on disk.
#
# Data: synthetic only, drawn with set.seed(20261007) BEFORE any sceptre call, because sceptre
# 0.10.3's get_random_starting_guesses() calls set.seed(4) and so resets the global stream.
# 30 genes x 1000 cells of gene expression, and 16 gRNA rows built to reach each path:
#   typ_5, typ_20     a perturbed subpopulation (5% and 20% of cells): the EM path
#   majority          70% perturbed: the EM flips its two components (pi > 0.5)
#   nz_9, nz_10       9 and 10 cells with a count >= 1: either side of the backup cutoff
#   zeros             no counts at all: the backup rule, assigning nothing
#   bg_only           background only
#   glm_extreme       one large count in 15 cells holding no other gRNA: glm.fit drives the other
#                     cells' fitted means to R's 2.2e-16 floor (pysceptre's IRLS floors at 1e-10),
#                     does not converge, and the EM fails, so the backup rule applies. sceptre's EM
#                     converged on every ordinary row tried, so this is the EM-failure path.
#   big               perturbed mean 2000: a cell's likelihood hits the 1e-100 floor
#   filler_1..5       typical rows
#
# Sections: provenance; starting_guesses; covariates (import_data's data frame), formula,
# design (column names and values); grnas (per gRNA: counts, the glm.fit fit, every EM start,
# the chosen start, the assignment); design_variants (the default design built on other
# covariate sets); public_equals_internal.
#
# Conventions: indices 0-based; a gRNA row is {j, x} over its nonzero cells; dense per-cell
# vectors in cell order; non-finite numbers are "Inf", "-Inf", "NaN", "NA".

suppressPackageStartupMessages({
  library(Matrix)
  library(sceptre)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("usage: dump_mixture_ground_truth.R <out.json>")
out_path <- args[[1]]
script_path <- normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)))

`%||%` <- function(a, b) if (is.null(a)) b else a
arr <- function(x) I(x)
num <- function(x) {
  x <- as.numeric(x)
  out <- as.list(x)
  out[is.na(x) & !is.nan(x)] <- "NA"
  out[is.nan(x)] <- "NaN"
  out[is.infinite(x) & x > 0] <- "Inf"
  out[is.infinite(x) & x < 0] <- "-Inf"
  out
}
scalar <- function(x) num(x)[[1]]
pkg <- function(p) {
  d <- utils::packageDescription(p)
  list(version = d$Version, remote_sha = d$RemoteSha %||% NA_character_)
}

N_CELLS <- 1000L
N_GENES <- 30L
# One cell with no gRNA UMI at all, so the gRNA covariates enter as log(x + 1); and the 15 cells
# of glm_extreme's face (see below).
zero_cell <- 7L
face <- seq(N_CELLS - 14L, N_CELLS)

import <- function(response_matrix, grna_matrix, extra = data.frame()) {
  suppressMessages(import_data(
    response_matrix = response_matrix,
    grna_matrix = grna_matrix,
    grna_target_data_frame = data.frame(grna_id = rownames(grna_matrix), grna_target = "non-targeting"),
    moi = "high",
    extra_covariates = extra
  ))
}

em_record <- function(g, mu0, sg) {
  lgf <- lgamma(g + 1)
  fit <- sceptre:::run_reduced_em_algo_cpp(sg$pi_guesses, sg$g_pert_guesses, g, mu0, lgf)
  starts <- lapply(seq_along(sg$pi_guesses), function(b) {
    f1 <- sceptre:::run_reduced_em_algo_cpp(sg$pi_guesses[b], sg$g_pert_guesses[b], g, mu0, lgf)
    list(converged = f1$outer_converged, log_lik = scalar(f1$outer_log_lik))
  })
  list(fit = fit, starts = starts)
}

grna_record <- function(name, g, X, sg) {
  n_nonzero <- sum(g >= 1)
  rec <- list(
    grna_id = name, n_nonzero = n_nonzero,
    j = arr(as.integer(which(g != 0) - 1L)), x = arr(as.numeric(g[g != 0]))
  )
  if (n_nonzero < 10) {
    rec$path <- "backup_n_nonzero"
  } else {
    pois <- withCallingHandlers(
      stats::glm.fit(y = g, x = X, family = stats::poisson()),
      warning = function(w) invokeRestart("muffleWarning")
    )
    mu0 <- pois$fitted.values
    em <- em_record(g, mu0, sg)
    fit <- em$fit
    rec$glm <- list(
      fitted = num(mu0), coefficients = num(pois$coefficients), converged = pois$converged,
      iter = pois$iter, deviance = scalar(pois$deviance), min_fitted = scalar(min(mu0))
    )
    rec$em <- list(
      outer_ti1s = num(fit$outer_Ti1s), outer_i = as.integer(fit$outer_i),
      outer_converged = fit$outer_converged, outer_log_lik = scalar(fit$outer_log_lik),
      starts = em$starts
    )
    rec$path <- if (fit$outer_converged && fit$outer_log_lik != -Inf) "mixture" else "backup_em"
    if (rec$path == "mixture") {
      rec$min_abs_ti1_minus_threshold <- scalar(min(abs(fit$outer_Ti1s - 0.8)))
      lls <- vapply(em$starts, function(s) if (isTRUE(s$converged) && is.numeric(s$log_lik)) s$log_lik else -Inf, numeric(1))
      lls <- sort(lls[is.finite(lls)], decreasing = TRUE)
      rec$start_gap <- if (length(lls) >= 2) scalar(lls[1] - lls[2]) else "NA"
    }
  }
  internal <- sceptre:::obtain_em_assignments(
    pi_guesses = sg$pi_guesses, g_pert_guesses = sg$g_pert_guesses, g = g, covariate_matrix = X,
    use_glm = TRUE, n_nonzero_cells_cutoff = 10L, backup_threshold = 5L, probability_threshold = 0.8
  )
  rec$assigned <- arr(as.integer(internal - 1L))
  rec
}

# One attempt at the data under a seed. Every random draw happens before any sceptre call, since
# sceptre 0.10.3's get_random_starting_guesses() calls set.seed(4). Returns NULL unless the
# invariants hold: the extreme row really reaches R's floor and fails the EM, nz_9 and nz_10 have
# 9 and 10 nonzero cells, and no EM-path cell sits within 1e-3 of the 0.8 cut, so a last-bit
# difference cannot flip an assignment.
attempt <- function(seed) {
  set.seed(seed)
  cell_size <- exp(rnorm(N_CELLS, 0, 0.4))
  gene_mu <- exp(rnorm(N_GENES, 1, 1))
  response <- matrix(rpois(N_GENES * N_CELLS, outer(gene_mu, cell_size)), N_GENES, N_CELLS)
  response[1, ] <- response[1, ] + 1L # every cell has at least one UMI
  rownames(response) <- paste0("gene_", seq_len(N_GENES))
  colnames(response) <- paste0("cell_", seq_len(N_CELLS))

  grna_depth <- exp(rnorm(N_CELLS, 0, 0.5))
  background <- function() rpois(N_CELLS, 0.15 * grna_depth)
  perturbed <- function(frac, mu) {
    p <- runif(N_CELLS) < frac
    g <- background()
    g[p] <- g[p] + rnbinom(sum(p), mu = mu * grna_depth[p], size = 5)
    g
  }
  exactly_nonzero <- function(k) {
    g <- integer(N_CELLS)
    g[sample(setdiff(seq_len(N_CELLS), c(zero_cell, face)), k)] <- 1L + rpois(k, 3)
    g
  }
  rows <- list(
    typ_5 = perturbed(0.05, 40),
    typ_20 = perturbed(0.20, 25),
    majority = perturbed(0.70, 30),
    nz_9 = exactly_nonzero(9),
    nz_10 = exactly_nonzero(10),
    zeros = integer(N_CELLS),
    bg_only = background(),
    big = perturbed(0.05, 2000),
    filler_1 = perturbed(0.03, 50),
    filler_2 = perturbed(0.08, 15),
    filler_3 = perturbed(0.12, 60),
    filler_4 = perturbed(0.02, 80),
    filler_5 = perturbed(0.10, 10),
    filler_6 = perturbed(0.06, 35)
  )
  extra <- data.frame(
    batch = sample(c("lane_a", "lane_b", "lane_c"), 200, TRUE),
    num_kept = sample(1:10, 200, TRUE) / 2,
    num_dropped = rnorm(200)
  )
  # glm_extreme: the same large count in 15 cells that hold no other gRNA, so those cells share
  # one value of every gRNA covariate and have the most gRNA UMIs of all. The zero cells are then
  # separated from them along log(grna_n_umis + 1), and glm.fit drives their fitted means towards
  # zero until it stops (pysceptre's IRLS floors a mean at 1e-10, R's at 2.2e-16).
  rows <- lapply(rows, function(g) {
    g[face] <- 0L
    g
  })
  glm_extreme <- integer(N_CELLS)
  glm_extreme[face] <- 5000L
  rows <- c(rows[setdiff(names(rows), "filler_6")], list(glm_extreme = glm_extreme, filler_6 = rows$filler_6))
  grna_matrix <- do.call(rbind, rows)
  grna_matrix[, zero_cell] <- 0L
  rownames(grna_matrix) <- names(rows)
  colnames(grna_matrix) <- colnames(response)

  sg <- sceptre:::get_random_starting_guesses(5L, c(1e-5, 0.1), log(c(10, 5000)))
  so <- import(response, grna_matrix)
  cov_df <- so@covariate_data_frame
  formula <- sceptre:::auto_construct_formula_object(cov_df, include_grna_covariates = TRUE)
  X <- sceptre:::convert_covariate_df_to_design_matrix(cov_df, formula)
  grnas <- lapply(rownames(grna_matrix), function(name) grna_record(name, as.numeric(grna_matrix[name, ]), X, sg))
  names(grnas) <- rownames(grna_matrix)

  ok <- grnas$glm_extreme$glm$min_fitted < 1e-10 && grnas$glm_extreme$path == "backup_em" &&
    grnas$nz_9$n_nonzero == 9 && grnas$nz_10$n_nonzero == 10 &&
    all(vapply(grnas, function(r) !identical(r$path, "mixture") || r$min_abs_ti1_minus_threshold >= 1e-3, logical(1)))
  if (!ok) return(NULL)
  list(seed = seed, response = response, grna_matrix = grna_matrix, extra = extra, sg = sg, so = so,
       cov_df = cov_df, formula = formula, X = X, grnas = grnas)
}

seeds <- 20261007L + 0:39
a <- NULL
for (s in seeds) {
  a <- attempt(s)
  if (!is.null(a)) break
}
if (is.null(a)) stop("no seed in the list satisfies the invariants")
response <- a$response
grna_matrix <- a$grna_matrix
sg <- a$sg
so <- a$so
cov_df <- a$cov_df
formula <- a$formula
X <- a$X
grnas <- a$grnas
batch <- a$extra$batch
num_kept <- a$extra$num_kept
num_dropped <- a$extra$num_dropped

# The public path, as the fishash paper's bin/run_sceptre_mixture.R calls it.
so_pub <- suppressMessages(so |> set_analysis_parameters() |> assign_grnas(method = "mixture", parallel = FALSE))
pub <- get_grna_assignments(so_pub)[rownames(grna_matrix), ]
public_equals_internal <- all(vapply(rownames(grna_matrix), function(name) {
  identical(as.integer(which(pub[name, ]) - 1L), as.integer(grnas[[name]]$assigned))
}, logical(1)))
stopifnot(public_equals_internal)

design_record <- function(label, cov) {
  f <- sceptre:::auto_construct_formula_object(cov, include_grna_covariates = TRUE)
  Xv <- sceptre:::convert_covariate_df_to_design_matrix(cov, f)
  list(label = label, formula = paste(deparse(f), collapse = " "), columns = arr(colnames(Xv)),
       X = lapply(seq_len(ncol(Xv)), function(k) num(Xv[, k])), covariates = lapply(cov, function(v) if (is.numeric(v)) num(v) else arr(as.character(v))))
}

sub_cells <- seq_len(200)
resp_mt <- response[, sub_cells]
rownames(resp_mt)[1:3] <- c("MT-CO1", "MT-ND1", "MT-ATP6")
grna_nozero <- grna_matrix[, sub_cells]
grna_nozero[1, ] <- grna_nozero[1, ] + 1L
so_logx <- import(response[, sub_cells], grna_nozero)
so_extra <- import(resp_mt, grna_matrix[, sub_cells],
                   extra = data.frame(batch = batch, num_kept = num_kept, num_dropped = num_dropped))
red_formula <- stats::formula(~ log(grna_n_nonzero + 1) + log(grna_n_umis + 1))
X_red <- sceptre:::convert_covariate_df_to_design_matrix(cov_df, red_formula)

out <- list(
  provenance = list(
    sceptre = pkg("sceptre"), Matrix = pkg("Matrix"), jsonlite = pkg("jsonlite"),
    r_version = R.version.string, platform = R.version$platform,
    sizeof_longdouble = .Machine$sizeof.longdouble,
    dumper_md5 = unname(tools::md5sum(script_path)),
    seed = a$seed
  ),
  starting_guesses = list(pi = num(sg$pi_guesses), g_pert = num(sg$g_pert_guesses)),
  n_cells = N_CELLS,
  grna_ids = arr(rownames(grna_matrix)),
  covariates = lapply(cov_df, num),
  formula = paste(deparse(formula), collapse = " "),
  design_columns = arr(colnames(X)),
  design = lapply(seq_len(ncol(X)), function(k) num(X[, k])),
  grnas = unname(grnas),
  public_equals_internal = public_equals_internal,
  design_variants = list(
    design_record("logx", so_logx@covariate_data_frame),
    design_record("extra", so_extra@covariate_data_frame),
    list(label = "reduced", formula = paste(deparse(red_formula), collapse = " "),
         columns = arr(colnames(X_red)), X = lapply(seq_len(ncol(X_red)), function(k) num(X_red[, k])))
  )
)

json <- jsonlite::toJSON(out, auto_unbox = TRUE, digits = I(17), na = "string", null = "null")
writeLines(json, out_path)
cat("wrote", out_path, format(file.size(out_path), big.mark = ","), "bytes\n")
