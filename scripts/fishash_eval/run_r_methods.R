#!/usr/bin/env Rscript
# Run the fishash comparison's R gRNA assignment methods on simulated datasets, timed.
#
# Usage:
#   Rscript scripts/fishash_eval/run_r_methods.R <sims_dir> <scenario> <runs_dir> <method> \
#     [--internals] [sim_label ...]
#
# <method> is one of
#   fishash_refit0, fishash_refit10
#       fishash::fishash(counts, refit = 0 or 10, padj_cutoff = 0.05, exclude_empty = TRUE), as
#       bin/run_fishash.R calls it. A second, untimed run traces fishash_internal() to record
#       what each pass cut and assigned, and must assign exactly what the timed run did.
#   sceptre_mixture
#       assign_grnas(method = "mixture", probability_threshold = 0.8, parallel = FALSE) on the
#       sceptre object bin/run_sceptre_mixture.R builds, retried with that script's reduced
#       formula if it errors. The gene expression is regenerated with gex_covariates.R's
#       functions and must match what that script recorded, so run it on the dataset first.
#       --internals then repeats sceptre's per-gRNA mixture fit outside sceptre, untimed, to
#       record what each fit did, and requires it to assign what assign_grnas() assigned.
#
# Outputs, under <runs_dir>/<scenario>/<sim_label>/r_<method>/:
#   assigned.parquet    (guide, cell) of every assignment, 0-based int32, column-major
#   log_pval.parquet    fishash: (guide, cell, log_pval) at every stored entry of the final
#                       pass's log p-values, column-major
#   per_grna.parquet    --internals: one row per gRNA, see mixture_internals()
#   ti1s_band.parquet   --internals: (guide, cell, ti1) where 0.5 <= Ti1 <= 0.95, on the gRNAs
#                       the mixture assigned, column-major
#   run.json            status, timings, settings, checks, thread variables and versions
# "seconds" times the assignment call alone; loading, regenerating the gex and building the
# sceptre object are timed separately. A run whose run.json says "ok" is skipped, unless
# --internals is asked of a sceptre_mixture run made without it. A failed run writes run.json
# with status "error", and the script exits non-zero once the other datasets are done.

# Attached up front, as bin/run_fishash.R does, so no namespace loads inside a timed call.
suppressPackageStartupMessages({
  library(Matrix)
  library(SummarizedExperiment)
  library(fishash)
})

script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "eval_common.R"))
# Function definitions only: its main() runs when it is the script Rscript was given.
source(file.path(script_dir, "gex_covariates.R"))

usage <- "usage: run_r_methods.R <sims_dir> <scenario> <runs_dir> <method> [--internals] [sim_label ...]"
known_methods <- c("fishash_refit0", "fishash_refit10", "sceptre_mixture")
output_files <- c("assigned.parquet", "log_pval.parquet", "per_grna.parquet", "ti1s_band.parquet", "run.json")
padj_cutoff <- 0.05
probability_threshold <- 0.8

thread_env <- function() {
  vars <- c("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
  stats::setNames(lapply(vars, function(v) {
    value <- Sys.getenv(v, unset = NA)
    if (is.na(value)) NULL else value
  }), vars)
}

machine <- function() {
  info <- Sys.info()
  list(sysname = info[["sysname"]], release = info[["release"]], machine = info[["machine"]],
       cores = parallel::detectCores())
}

# Elapsed and CPU seconds of evaluating `expr` in the caller's frame, after a gc().
timed <- function(expr) {
  t <- system.time(expr)
  list(seconds = t[["elapsed"]], cpu_seconds = t[["user.self"]] + t[["sys.self"]])
}

write_parquet_atomic <- function(df, path) write_atomic(path, function(tmp) arrow::write_parquet(df, tmp))

# A sparse matrix's stored entries as (guide, cell, <name>), 0-based, column-major.
value_triplets <- function(m, name) {
  m <- as(m, "CsparseMatrix")
  if (!is(m, "dgCMatrix")) stop("expected a dgCMatrix, got ", class(m)[[1]])
  out <- data.frame(guide = as.integer(m@i), cell = rep.int(seq_len(ncol(m)) - 1L, diff(m@p)))
  out[[name]] <- m@x
  out
}

# Logical or pattern matrix -> assigned.parquet.
write_assigned <- function(m, path) {
  write_triplets(as(as(m, "CsparseMatrix"), "lMatrix"), path, logical = TRUE)
}

require_checks <- function(checks) {
  failed <- names(checks)[!vapply(checks, isTRUE, logical(1))]
  if (length(failed) > 0L) stop("checks failed: ", paste(failed, collapse = ", "))
}

# ---- fishash --------------------------------------------------------------------------------

fishash_call <- function(counts, refit) {
  fishash::fishash(counts, refit = refit, padj_cutoff = padj_cutoff, exclude_empty = TRUE)
}

# fishash() with fishash_internal() traced: one record per pass, read from the pass's frame.
traced_fishash <- function(counts, refit) {
  passes <- list()
  record <- function(frame) {
    local_value <- function(name) get0(name, envir = frame, inherits = FALSE)
    passes[[length(passes) + 1L]] <<- list(
      pass = length(passes) + 1L,
      logpval_cutoff = local_value("logpval_cutoff"),
      B = local_value("B"),
      n_signif = local_value("n_signif"),
      n_assigned = sum(local_value("mat_assigned"))
    )
  }
  ns <- asNamespace("fishash")
  suppressMessages(trace("fishash_internal", where = ns, print = FALSE,
                         exit = bquote(.(record)(environment()))))
  on.exit(suppressMessages(untrace("fishash_internal", where = ns)), add = TRUE)
  res <- fishash_call(counts, refit)
  list(res = res, passes = passes)
}

run_fishash <- function(rds_path, refit, out_dir) {
  load <- timed(counts <- assay(readRDS(rds_path), "counts"))
  run <- timed(res <- fishash_call(counts, refit))
  traced_run <- timed(traced <- traced_fishash(counts, refit))
  assigned <- assay(res, "assigned")
  passes <- traced$passes
  checks <- list(
    untraced_after = !is(get("fishash_internal", envir = asNamespace("fishash")), "functionWithTrace"),
    traced_assigned_identical = identical(assay(traced$res, "assigned"), assigned),
    n_passes_equals_num_iter = length(passes) == metadata(res)$num_iter,
    last_pass_cutoff_identical = length(passes) > 0L &&
      identical(passes[[length(passes)]]$logpval_cutoff, metadata(res)$log_pval_cutoff)
  )
  require_checks(checks)

  write_assigned(assigned, file.path(out_dir, "assigned.parquet"))
  log_pval <- value_triplets(assay(res, "log_pval"), "log_pval")
  write_parquet_atomic(log_pval, file.path(out_dir, "log_pval.parquet"))

  demux <- table(factor(colData(res)$demux_type, levels = c("singlet", "doublet", "unknown")))
  defaults <- formals(fishash::fishash)
  list(
    seconds = run$seconds,
    cpu_seconds = run$cpu_seconds,
    load_seconds = load$seconds,
    traced_seconds = traced_run$seconds,
    n_guides = nrow(counts),
    n_cells = ncol(counts),
    cutoff = metadata(res)$log_pval_cutoff,
    num_iter = metadata(res)$num_iter,
    n_assigned = sum(assigned),
    demux_type = as.list(stats::setNames(as.integer(demux), names(demux))),
    n_log_pval_entries = nrow(log_pval),
    passes = passes,
    checks = checks,
    settings = list(
      call = sprintf("fishash::fishash(counts, refit = %d, padj_cutoff = %s, exclude_empty = TRUE)",
                     refit, format(padj_cutoff)),
      refit = refit,
      padj_cutoff = padj_cutoff,
      exclude_empty = TRUE,
      defaults_not_passed = list(
        padj_method = eval(defaults$padj_method)[[1]],
        min_count = defaults$min_count,
        min_frac = defaults$min_frac
      ),
      upstream_script = "bin/run_fishash.R"
    ),
    versions = versions()
  )
}

# ---- sceptre mixture ------------------------------------------------------------------------

covariates_identical <- function(observed, parquet_path) {
  recorded <- as.data.frame(arrow::read_parquet(parquet_path))
  if (!identical(names(recorded)[1], "cell")) return(FALSE)
  if (!identical(as.vector(recorded$cell), seq_len(nrow(observed)) - 1L)) return(FALSE)
  recorded$cell <- NULL
  identical(names(recorded), names(observed)) &&
    all(vapply(names(observed), function(n) identical(as.vector(recorded[[n]]), as.vector(observed[[n]])),
               logical(1)))
}

# sceptre's obtain_em_assignments(), one gRNA at a time, keeping what each fit did. The formula,
# starting guesses and cutoffs are the ones assign_grnas() used. Only the current gRNA's row is
# made dense, by sceptre's own load_row().
#
# per_grna: guide (0-based); n_nonzero, cells with a count; path, "mixture" when the EM's best
# start converged with a finite log-likelihood, else "backup" (count >= backup_threshold);
# glm_converged, glm_iter of the Poisson GLM; outer_converged, outer_i (0-based start index),
# outer_log_lik as run_reduced_em_algo_cpp() returns them; start_gap, the best converged
# start's log-likelihood minus the second best's, NA with fewer than two;
# n_starts_converged; n_assigned. The GLM and EM columns are NA below n_nonzero_cells_cutoff.
mixture_internals <- function(sceptre_object, guide_counts, assignments) {
  hp <- sceptre_object@grna_assignment_hyperparameters
  covariate_matrix <- sceptre:::convert_covariate_df_to_design_matrix(
    covariate_data_frame = sceptre_object@covariate_data_frame,
    formula_object = hp$formula_object
  )
  grna_matrix <- sceptre:::set_matrix_accessibility(sceptre:::get_grna_matrix(sceptre_object),
                                                    make_row_accessible = TRUE)
  # get_random_starting_guesses() calls set.seed(4), so nothing random may come after it.
  sg <- sceptre:::get_random_starting_guesses(
    n_em_rep = hp$n_em_rep,
    pi_guess_range = hp$pi_guess_range,
    g_pert_guess_range = hp$g_pert_guess_range
  )
  rows <- as(assignments, "RsparseMatrix")
  grna_ids <- rownames(guide_counts)
  if (!identical(rownames(rows), grna_ids)) stop("assignment rows are not in the count matrix's order")
  n <- length(grna_ids)

  n_nonzero <- integer(n)
  path <- character(n)
  glm_converged <- rep(NA, n)
  glm_iter <- rep(NA_integer_, n)
  outer_converged <- rep(NA, n)
  outer_i <- rep(NA_integer_, n)
  outer_log_lik <- rep(NA_real_, n)
  start_gap <- rep(NA_real_, n)
  n_starts_converged <- rep(NA_integer_, n)
  n_assigned <- integer(n)
  matches <- logical(n)
  best_start_consistent <- rep(NA, n)
  band_guide <- vector("list", n)
  band_cell <- vector("list", n)
  band_ti1 <- vector("list", n)

  for (k in seq_len(n)) {
    g <- sceptre:::load_row(grna_matrix, grna_ids[[k]])
    n_nonzero[k] <- sum(g >= 1)
    assigned <- NULL
    if (n_nonzero[k] >= hp$n_nonzero_cells_cutoff) {
      pois_fit <- suppressWarnings(stats::glm.fit(y = g, x = covariate_matrix, family = stats::poisson()))
      mu0 <- pois_fit$fitted.values
      log_g_factorial <- lgamma(g + 1)
      fit <- sceptre:::run_reduced_em_algo_cpp(sg$pi_guesses, sg$g_pert_guesses, g, mu0, log_g_factorial)
      starts <- lapply(seq_along(sg$pi_guesses), function(s) {
        sceptre:::run_reduced_em_algo_cpp(sg$pi_guesses[s], sg$g_pert_guesses[s], g, mu0, log_g_factorial)
      })
      start_converged <- vapply(starts, function(f) f$outer_converged, logical(1))
      start_log_lik <- vapply(starts, function(f) f$outer_log_lik, numeric(1))
      ranked <- sort(start_log_lik[start_converged], decreasing = TRUE)

      glm_converged[k] <- pois_fit$converged
      glm_iter[k] <- pois_fit$iter
      outer_converged[k] <- fit$outer_converged
      outer_i[k] <- fit$outer_i
      outer_log_lik[k] <- fit$outer_log_lik
      n_starts_converged[k] <- sum(start_converged)
      if (length(ranked) >= 2L) start_gap[k] <- ranked[[1]] - ranked[[2]]
      # The C++ keeps the first start with the strictly highest converged log-likelihood.
      best_start_consistent[k] <- if (any(start_converged)) {
        best <- which(start_converged & start_log_lik == ranked[[1]])[[1]]
        identical(fit$outer_log_lik, start_log_lik[[best]]) && fit$outer_i == best - 1L &&
          identical(fit$outer_Ti1s, starts[[best]]$outer_Ti1s)
      } else {
        !fit$outer_converged
      }

      # obtain_em_assignments()'s own test.
      if (fit$outer_converged && fit$outer_log_lik != -Inf) {
        assigned <- which(fit$outer_Ti1s >= hp$probability_threshold)
        in_band <- which(fit$outer_Ti1s >= 0.5 & fit$outer_Ti1s <= 0.95)
        band_guide[[k]] <- rep.int(k - 1L, length(in_band))
        band_cell[[k]] <- in_band - 1L
        band_ti1[[k]] <- fit$outer_Ti1s[in_band]
      }
    }
    path[k] <- if (is.null(assigned)) "backup" else "mixture"
    if (is.null(assigned)) assigned <- which(g >= hp$backup_threshold)
    n_assigned[k] <- length(assigned)
    from_sceptre <- rows@j[seq.int(rows@p[k] + 1L, length.out = rows@p[k + 1L] - rows@p[k])] + 1L
    matches[k] <- identical(sort(as.integer(assigned)), sort(as.integer(from_sceptre)))
  }

  band <- data.frame(guide = as.integer(unlist(band_guide)), cell = as.integer(unlist(band_cell)),
                     ti1 = as.numeric(unlist(band_ti1)))
  band <- band[order(band$cell, band$guide), , drop = FALSE]
  rownames(band) <- NULL
  list(
    per_grna = data.frame(
      guide = seq_len(n) - 1L, n_nonzero = n_nonzero, path = path,
      glm_converged = glm_converged, glm_iter = glm_iter,
      outer_converged = outer_converged, outer_i = outer_i, outer_log_lik = outer_log_lik,
      start_gap = start_gap, n_starts_converged = n_starts_converged, n_assigned = n_assigned,
      stringsAsFactors = FALSE
    ),
    band = band,
    summary = list(
      n_grnas = n,
      n_mixture = sum(path == "mixture"),
      n_backup_below_cutoff = sum(n_nonzero < hp$n_nonzero_cells_cutoff),
      n_backup_em_rejected = sum(path == "backup" & n_nonzero >= hp$n_nonzero_cells_cutoff),
      n_glm_not_converged = sum(glm_converged %in% FALSE),
      n_band_entries = nrow(band),
      assignments_match_all = all(matches),
      n_mismatched = sum(!matches),
      mismatched_head = utils::head(grna_ids[!matches], 20L),
      best_start_consistent = list(n_checked = sum(!is.na(best_start_consistent)),
                                   n_consistent = sum(best_start_consistent, na.rm = TRUE)),
      starting_guesses = list(pi = sg$pi_guesses, g_pert = sg$g_pert_guesses)
    )
  )
}

run_sceptre_mixture <- function(d, rds_path, sim_dir, out_dir, internals) {
  cov_json <- file.path(sim_dir, "covariates.json")
  cov_parquet <- file.path(sim_dir, "covariates.parquet")
  if (!file.exists(cov_json) || !file.exists(cov_parquet)) {
    stop("no covariates for ", d$sim_label, "; run gex_covariates.R on it first")
  }
  recorded <- jsonlite::read_json(cov_json)

  load <- timed(guide_counts <- load_guide_counts(rds_path))
  gex_time <- timed(gex <- regenerate_gex(guide_counts, d$rds))
  invisible(gc())
  checks <- list(gex_sha256_identical = identical(gex_sha256(gex), recorded$gex$sha256))
  require_checks(checks)

  # bin/run_sceptre_mixture.R seeds itself before building the object. Nothing after this
  # draws before assign_grnas(), which calls set.seed(4) itself, but the state is the same.
  run_seed_string <- paste0("run_sceptre_mixture.R", "_", d$rds)
  set.seed(abs(digest::digest2int(run_seed_string)))
  import <- timed({
    sceptre_object <- import_like_upstream(gex, guide_counts)
    sceptre_object <- set_analysis_parameters(sceptre_object)
  })
  rm(gex)
  invisible(gc())
  covariates <- sceptre_object@covariate_data_frame
  checks$covariates_identical <- covariates_identical(covariates, cov_parquet)
  checks$default_formula_identical <- identical(formula_text(default_mixture_formula(covariates)),
                                                recorded$default_mixture_formula)
  require_checks(checks)

  fallback_used <- FALSE
  fallback_error <- NULL
  run <- timed(sceptre_object <- tryCatch(
    assign_grnas(sceptre_object, method = "mixture", probability_threshold = probability_threshold,
                 parallel = FALSE),
    error = function(e) {
      fallback_used <<- TRUE
      fallback_error <<- conditionMessage(e)
      print("Failed, trying with reduced design")
      assign_grnas(sceptre_object, method = "mixture", probability_threshold = probability_threshold,
                   formula_object = formula(~ log(grna_n_nonzero+1) + log(grna_n_umis+1)),
                   parallel = FALSE)
    }
  ))
  hp <- sceptre_object@grna_assignment_hyperparameters
  assignments <- get_grna_assignments(sceptre_object = sceptre_object)[rownames(guide_counts), ]
  colnames(assignments) <- colnames(guide_counts)
  write_assigned(assignments, file.path(out_dir, "assigned.parquet"))

  internals_record <- list(requested = internals)
  if (internals) {
    internals_time <- timed(found <- mixture_internals(sceptre_object, guide_counts, assignments))
    write_parquet_atomic(found$per_grna, file.path(out_dir, "per_grna.parquet"))
    write_parquet_atomic(found$band, file.path(out_dir, "ti1s_band.parquet"))
    internals_record <- c(internals_record, list(seconds = internals_time$seconds), found$summary)
    checks$internals_assignments_identical <- found$summary$assignments_match_all
    require_checks(checks)
  }

  list(
    seconds = run$seconds,
    cpu_seconds = run$cpu_seconds,
    load_seconds = load$seconds,
    gex_seconds = gex_time$seconds,
    import_seconds = import$seconds,
    n_guides = nrow(guide_counts),
    n_cells = ncol(guide_counts),
    fallback_used = fallback_used,
    fallback_error = fallback_error,
    formula = formula_text(hp$formula_object),
    n_assigned = length(as(assignments, "CsparseMatrix")@i),
    checks = checks,
    internals = internals_record,
    settings = list(
      call = sprintf("assign_grnas(sceptre_object, method = \"mixture\", probability_threshold = %s, parallel = FALSE)",
                     format(probability_threshold)),
      fallback_formula = "~log(grna_n_nonzero + 1) + log(grna_n_umis + 1)",
      hyperparameters = hp[setdiff(names(hp), "formula_object")],
      gex_seed_string = gex_seed_string(d$rds),
      gex_seed = gex_seed(d$rds),
      run_seed_string = run_seed_string,
      run_seed = abs(digest::digest2int(run_seed_string)),
      upstream_scripts = c("bin/get_gex_matrix.R", "bin/run_sceptre_mixture.R")
    ),
    versions = versions(c("sceptre", "splatter", "fishash", "Matrix", "SummarizedExperiment",
                          "SingleCellExperiment", "digest"))
  )
}

# ---- main -----------------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4L) stop(usage)
sims_dir <- args[[1]]
scenario <- args[[2]]
runs_dir <- args[[3]]
method <- args[[4]]
rest <- args[-(1:4)]
is_flag <- startsWith(rest, "--")
unknown_flags <- setdiff(rest[is_flag], "--internals")
if (length(unknown_flags) > 0L) stop("unknown option ", paste(unknown_flags, collapse = ", "), "\n", usage)
internals <- "--internals" %in% rest[is_flag]
if (!(method %in% known_methods)) stop("method must be one of ", paste(known_methods, collapse = ", "))
if (internals && method != "sceptre_mixture") stop("--internals applies to sceptre_mixture only")

scenario_dir <- file.path(sims_dir, scenario)
manifest <- read_manifest(scenario_dir)
datasets <- select_datasets(manifest, rest[!is_flag])
refuse_unless_ignored(file.path(runs_dir, scenario))

run_is_done <- function(run_json) {
  if (!file.exists(run_json)) return(FALSE)
  run <- jsonlite::read_json(run_json)
  identical(run$status, "ok") && (!internals || isTRUE(run$internals$requested))
}

failed <- character()
for (d in datasets) {
  out_dir <- file.path(runs_dir, scenario, d$sim_label, paste0("r_", method))
  run_json <- file.path(out_dir, "run.json")
  if (run_is_done(run_json)) {
    stamp("skip ", method, " on ", d$sim_label, ": run.json says ok")
    next
  }
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  unlink(file.path(out_dir, output_files))
  rds_path <- file.path(scenario_dir, "rds", d$rds)
  header <- list(
    status = "ok",
    method = method,
    implementation = "R",
    scenario = scenario,
    sim_label = d$sim_label,
    rds = d$rds,
    rds_sha256 = sha256_file(rds_path),
    threads = thread_env(),
    machine = machine()
  )
  stamp(method, " on ", d$sim_label)
  result <- tryCatch(
    switch(method,
      fishash_refit0 = run_fishash(rds_path, 0L, out_dir),
      fishash_refit10 = run_fishash(rds_path, 10L, out_dir),
      sceptre_mixture = run_sceptre_mixture(d, rds_path, file.path(scenario_dir, d$sim_label), out_dir, internals)
    ),
    error = function(e) e
  )
  if (inherits(result, "error")) {
    header$status <- "error"
    write_json_atomic(c(header, list(error = conditionMessage(result), versions = versions())), run_json)
    stamp("FAILED ", method, " on ", d$sim_label, ": ", conditionMessage(result))
    failed <- c(failed, d$sim_label)
  } else {
    write_json_atomic(c(header, result, list(finished = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"))), run_json)
    stamp("done ", method, " on ", d$sim_label, ": ", result$n_assigned, " assigned in ",
          round(result$seconds, 2), " s")
  }
}
if (length(failed) > 0L) {
  stamp(length(failed), " failed: ", paste(failed, collapse = ", "))
  quit(status = 1)
}
stamp("done ", method, " on ", scenario, ": ", length(datasets), " datasets")
