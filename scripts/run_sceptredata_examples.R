#!/usr/bin/env Rscript
# R reference analyses of sceptredata's two real example screens.
#
# Usage:
#   Rscript scripts/run_sceptredata_examples.R <out_dir> [lowmoi|highmoi|all]
#
# Datasets (sceptredata 0.99.0, loaded with data(..., package = "sceptredata")):
#   lowmoi   lowmoi_example_data, Papalexi 2021 CRISPRko, 299 genes x 20,729 cells
#   highmoi  highmoi_example_data + grna_target_data_frame_highmoi, Gasperini 2019
#            CRISPRi, 526 genes x 45,919 cells
# sceptre 0.10.3 ships simulated datasets under the same names, so the data are
# always read from the sceptredata package explicitly, never from the search path.
#
# Runs:
#   lowmoi   control_group in {complement, nt_cells} x resampling_mechanism in {crt, permutations}
#   highmoi  complement x {crt, permutations} (the only control group in high MOI)
#
# Pipeline, per run, as documented in sceptre v0.10.1's R/sceptre.R @examples (the last
# release whose examples used sceptredata), with the high-MOI import taken from the
# import_data() route of the 0.10.x function @examples:
#   lowmoi   import_data(moi = "low", extra_covariates) -> construct_positive_control_pairs()
#            -> construct_trans_pairs(pairs_to_exclude = "pc_pairs") -> side = "both"
#   highmoi  import_data(moi = "high", extra_covariates, response_names = gene_names)
#            -> construct_positive_control_pairs() -> construct_cis_pairs(distance_threshold = 5e6)
#            -> side = "left"
#   both     set_analysis_parameters() -> assign_grnas() (default method) ->
#            run_qc(p_mito_threshold = 0.075) -> run_calibration_check() -> run_power_check()
#            -> run_discovery_analysis(), all with parallel = FALSE and output_amount = 2
#            (adds stage, z_orig, xi, omega, alpha; the computed values are unchanged).
#
# Output, per run, in <out_dir>/<dataset>/<control_group>_<resampling_mechanism>/:
#   sceptre_object.rds           the analysed object (exact values, gRNA assignments,
#                                cells_in_use, negative_control_pairs)
#   run_calibration_check.csv    every column of get_result(); doubles written with 17
#   run_power_check.csv          significant digits, exact under a correctly rounded parser
#   run_discovery_analysis.csv   (Python float(), pandas.read_csv(float_precision="round_trip")).
#                                pandas' default parser and R's read.csv are not correctly
#                                rounded and can be off by a few ulp.
#   run_info.json                versions, settings read back from the object, wall times,
#                                result counts, warnings and errors
#   log.txt                      console output of the run
# A run whose run_info.json has status "ok" and whose files all exist is skipped.
#
# sceptredata is real published data. The script refuses an out_dir inside a git work
# tree unless git ignores it; test_data/ is ignored in this repository.

suppressPackageStartupMessages({
  library(sceptre)
  library(Matrix)
})

SEED <- 4L
P_MITO_THRESHOLD <- 0.075
DISTANCE_THRESHOLD <- 5e6
OUTPUT_AMOUNT <- 2L
ANALYSES <- c("run_calibration_check", "run_power_check", "run_discovery_analysis")
RUNS <- list(
  lowmoi = list(
    c(control_group = "complement", resampling_mechanism = "crt"),
    c(control_group = "complement", resampling_mechanism = "permutations"),
    c(control_group = "nt_cells", resampling_mechanism = "crt"),
    c(control_group = "nt_cells", resampling_mechanism = "permutations")
  ),
  highmoi = list(
    c(control_group = "complement", resampling_mechanism = "crt"),
    c(control_group = "complement", resampling_mechanism = "permutations")
  )
)

`%||%` <- function(a, b) if (is.null(a)) b else a

stamp <- function(...) cat(sprintf("[%s] ", format(Sys.time(), "%Y-%m-%d %H:%M:%S")), ..., "\n", sep = "")

refuse_unless_ignored <- function(out_dir) {
  git <- Sys.which("git")
  if (!nzchar(git)) return(invisible())
  anchor <- out_dir
  while (!dir.exists(anchor)) anchor <- dirname(anchor)
  in_tree <- suppressWarnings(system2(git, c("-C", shQuote(anchor), "rev-parse", "--is-inside-work-tree"),
                                      stdout = TRUE, stderr = FALSE))
  if (!identical(in_tree, "true")) return(invisible())
  probe <- file.path(out_dir, "probe", "run_info.json")
  ignored <- system2(git, c("-C", shQuote(anchor), "check-ignore", "-q", shQuote(probe)), stdout = FALSE, stderr = FALSE)
  if (ignored != 0L) {
    stop("refusing to write real screen data to ", out_dir, ": it is inside a git work tree and not ignored")
  }
}

load_sceptredata <- function() {
  e <- new.env()
  utils::data(list = c("lowmoi_example_data", "highmoi_example_data", "grna_target_data_frame_highmoi"),
              package = "sceptredata", envir = e)
  stopifnot(
    identical(dim(e$lowmoi_example_data$response_matrix), c(299L, 20729L)),
    identical(dim(e$highmoi_example_data$response_matrix), c(526L, 45919L)),
    nrow(e$grna_target_data_frame_highmoi) == 95L
  )
  e
}

dataset_spec <- function(dataset, data_env) {
  switch(dataset,
    lowmoi = list(
      description = "sceptredata lowmoi_example_data: Papalexi 2021 CRISPRko, THP-1",
      side = "both",
      import = function() {
        d <- data_env$lowmoi_example_data
        import_data(
          response_matrix = d$response_matrix,
          grna_matrix = d$grna_matrix,
          extra_covariates = d$extra_covariates,
          grna_target_data_frame = d$grna_target_data_frame,
          moi = "low"
        )
      },
      pairs = function(so) {
        pc <- construct_positive_control_pairs(so)
        disc <- construct_trans_pairs(so, positive_control_pairs = pc, pairs_to_exclude = "pc_pairs")
        list(positive_control_pairs = pc, discovery_pairs = disc)
      },
      pipeline = list(
        source = "sceptre v0.10.1 R/sceptre.R @examples, low-MOI CRISPRko example",
        import = "import_data(response_matrix, grna_matrix, extra_covariates, grna_target_data_frame, moi = 'low')",
        positive_control_pairs = "construct_positive_control_pairs(sceptre_object)",
        discovery_pairs = "construct_trans_pairs(sceptre_object, positive_control_pairs, pairs_to_exclude = 'pc_pairs')"
      )
    ),
    highmoi = list(
      description = "sceptredata highmoi_example_data + grna_target_data_frame_highmoi: Gasperini 2019 CRISPRi, K562",
      side = "left",
      import = function() {
        d <- data_env$highmoi_example_data
        import_data(
          response_matrix = d$response_matrix,
          grna_matrix = d$grna_matrix,
          grna_target_data_frame = data_env$grna_target_data_frame_highmoi,
          moi = "high",
          extra_covariates = d$extra_covariates,
          response_names = d$gene_names
        )
      },
      pairs = function(so) {
        pc <- construct_positive_control_pairs(so)
        disc <- construct_cis_pairs(so, positive_control_pairs = pc, distance_threshold = DISTANCE_THRESHOLD)
        list(positive_control_pairs = pc, discovery_pairs = disc)
      },
      pipeline = list(
        source = paste(
          "sceptre v0.10.1 R/sceptre.R @examples, high-MOI CRISPRi example; that example imports",
          "sceptredata's extdata with import_data_from_cellranger(), this uses the import_data()",
          "route of the v0.10.1 function @examples (e.g. set_analysis_parameters)"
        ),
        import = "import_data(response_matrix, grna_matrix, grna_target_data_frame_highmoi, moi = 'high', extra_covariates, response_names = gene_names)",
        positive_control_pairs = "construct_positive_control_pairs(sceptre_object)",
        discovery_pairs = "construct_cis_pairs(sceptre_object, positive_control_pairs, distance_threshold = 5e6)"
      )
    )
  )
}

# Runs f() with its wall time, warnings (muffled, tabulated) and messages recorded; an error is
# recorded rather than raised, and the value is then NULL.
run_step <- function(name, f) {
  warnings <- character()
  messages <- character()
  error <- NULL
  value <- NULL
  stamp("start ", name)
  elapsed <- system.time({
    value <- tryCatch(
      withCallingHandlers(f(),
        warning = function(w) {
          warnings <<- c(warnings, conditionMessage(w))
          invokeRestart("muffleWarning")
        },
        message = function(m) messages <<- c(messages, conditionMessage(m))
      ),
      error = function(e) {
        error <<- conditionMessage(e)
        NULL
      }
    )
  })[["elapsed"]]
  tab <- if (length(warnings)) sort(table(warnings), decreasing = TRUE) else integer()
  warning_list <- lapply(names(tab), function(m) list(message = m, count = as.integer(tab[[m]])))
  for (w in warning_list) cat(sprintf("  R warning [x%d]: %s\n", w$count, w$message))
  if (!is.null(error)) cat("  ERROR in ", name, ": ", error, "\n", sep = "")
  stamp(sprintf("end   %s (%.1f s)", name, elapsed))
  list(
    value = value,
    record = list(
      wall_time_s = elapsed,
      error = error,
      warnings = warning_list,
      messages = unique(trimws(messages))
    )
  )
}

plain <- function(x) {
  lapply(x, function(v) if (inherits(v, "formula")) paste(deparse(v), collapse = " ") else v)
}

object_summary <- function(so) {
  ga <- so@grna_assignments
  group_sizes <- lengths(ga$grna_group_idxs)
  nonzero_sizes <- group_sizes[group_sizes > 0L]
  n_nt_cells <- if (!is.null(ga$all_nt_idxs)) length(ga$all_nt_idxs) else length(unique(unlist(ga$indiv_nt_grna_idxs)))
  list(
    moi = if (so@low_moi) "low" else "high",
    control_group = if (so@control_group_complement) "complement" else "nt_cells",
    resampling_mechanism = if (so@run_permutations) "permutations" else "crt",
    side = c("left", "both", "right")[so@side_code + 2L],
    grna_integration_strategy = so@grna_integration_strategy,
    resampling_approximation = so@resampling_approximation,
    multiple_testing_method = so@multiple_testing_method,
    multiple_testing_alpha = so@multiple_testing_alpha,
    formula = paste(deparse(so@formula_object), collapse = " "),
    covariate_matrix_columns = colnames(so@covariate_matrix),
    grna_assignment_method = so@grna_assignment_method,
    grna_assignment_hyperparameters = plain(so@grna_assignment_hyperparameters),
    cellwise_qc_thresholds = so@cellwise_qc_thresholds,
    n_nonzero_trt_thresh = so@n_nonzero_trt_thresh,
    n_nonzero_cntrl_thresh = so@n_nonzero_cntrl_thresh,
    B1 = so@B1,
    B2 = so@B2,
    B3 = so@B3,
    calibration_group_size = if (length(so@calibration_group_size)) so@calibration_group_size else NA,
    n_calibration_pairs = if (length(so@n_calibration_pairs)) so@n_calibration_pairs else NA,
    n_negative_control_pairs = nrow(so@negative_control_pairs),
    n_cells = nrow(so@covariate_data_frame),
    n_cells_in_use = length(so@cells_in_use),
    cell_removal_metrics = as.list(so@cell_removal_metrics),
    n_nt_cells = n_nt_cells,
    indiv_nt_grna_n_cells = as.list(lengths(ga$indiv_nt_grna_idxs)),
    n_grna_groups = length(group_sizes),
    n_grna_groups_with_cells = length(nonzero_sizes),
    grna_group_n_cells_range = if (length(nonzero_sizes)) range(nonzero_sizes) else NA,
    n_discovery_pairs = so@n_discovery_pairs,
    n_ok_discovery_pairs = if (length(so@n_ok_discovery_pairs)) so@n_ok_discovery_pairs else NA,
    n_positive_control_pairs = so@n_positive_control_pairs,
    n_ok_positive_control_pairs = if (length(so@n_ok_positive_control_pairs)) so@n_ok_positive_control_pairs else NA
  )
}

result_summary <- function(df, file) {
  has <- function(col) col %in% names(df)
  ok <- if (has("pass_qc")) df$pass_qc %in% TRUE else rep(TRUE, nrow(df))
  list(
    file = file,
    columns = names(df),
    rows = nrow(df),
    pass_qc = if (has("pass_qc")) sum(df$pass_qc, na.rm = TRUE) else NA,
    significant = if (has("significant")) sum(df$significant, na.rm = TRUE) else NA,
    p_value_not_na = sum(!is.na(df$p_value)),
    median_p_value_pass_qc = if (any(ok)) stats::median(df$p_value[ok], na.rm = TRUE) else NA,
    stage_counts = if (has("stage")) as.list(table(stage = df$stage[ok], useNA = "ifany")) else NA
  )
}

write_atomic <- function(path, writer) {
  tmp <- paste0(path, ".tmp")
  writer(tmp)
  if (!file.rename(tmp, path)) stop("could not move ", tmp, " to ", path)
}

write_result_csv <- function(df, path) {
  df <- as.data.frame(df)
  quote_cols <- which(vapply(df, function(x) is.character(x) || is.factor(x), logical(1)))
  for (col in names(df)) {
    x <- df[[col]]
    if (is.double(x)) {
      s <- sprintf("%.17g", x)
      s[is.na(x)] <- NA_character_
      df[[col]] <- s
    }
  }
  write_atomic(path, function(tmp) {
    utils::write.csv(df, tmp, row.names = FALSE, na = "NA", quote = if (length(quote_cols)) quote_cols else FALSE)
  })
}

versions <- function() {
  ds <- utils::packageDescription("sceptre")
  dd <- utils::packageDescription("sceptredata")
  list(
    sceptre = ds$Version,
    sceptre_remote_sha = ds$RemoteSha %||% NA,
    sceptredata = dd$Version,
    sceptredata_remote_sha = dd$RemoteSha %||% NA,
    r = R.version.string,
    platform = R.version$platform,
    blas = utils::sessionInfo()$BLAS,
    lapack = La_library()
  )
}

expected_files <- function() c("sceptre_object.rds", paste0(ANALYSES, ".csv"), "run_info.json")

run_is_complete <- function(run_dir) {
  info_path <- file.path(run_dir, "run_info.json")
  if (!all(file.exists(file.path(run_dir, expected_files())))) return(FALSE)
  info <- tryCatch(jsonlite::read_json(info_path), error = function(e) NULL)
  identical(info$status, "ok")
}

run_one <- function(dataset, control_group, resampling_mechanism, data_env, out_dir) {
  run_name <- paste0(control_group, "_", resampling_mechanism)
  run_dir <- file.path(out_dir, dataset, run_name)
  if (run_is_complete(run_dir)) {
    stamp("skip ", dataset, "/", run_name, ": outputs exist")
    return("cached")
  }
  dir.create(run_dir, recursive = TRUE, showWarnings = FALSE)
  unlink(file.path(run_dir, expected_files()))
  log_path <- file.path(run_dir, "log.txt")
  log_con <- file(log_path, open = "wt")
  sink(log_con, split = TRUE)
  on.exit({
    sink()
    close(log_con)
  }, add = TRUE)

  stamp("run ", dataset, "/", run_name)
  set.seed(SEED)
  spec <- dataset_spec(dataset, data_env)
  steps <- list()
  so <- NULL
  pairs <- NULL

  preparation <- list(
    import_data = function(x) {
      x <- spec$import()
      if (!"response_p_mito" %in% colnames(x@covariate_data_frame)) {
        stop("no response_p_mito covariate: p_mito_threshold would have no effect")
      }
      x
    },
    construct_pairs = function(x) {
      pairs <<- spec$pairs(x)
      if (nrow(pairs$discovery_pairs) == 0L || nrow(pairs$positive_control_pairs) == 0L) {
        stop("empty pair set: ", nrow(pairs$discovery_pairs), " discovery, ",
             nrow(pairs$positive_control_pairs), " positive control")
      }
      x
    },
    set_analysis_parameters = function(x) {
      set_analysis_parameters(
        sceptre_object = x,
        discovery_pairs = pairs$discovery_pairs,
        positive_control_pairs = pairs$positive_control_pairs,
        side = spec$side,
        control_group = control_group,
        resampling_mechanism = resampling_mechanism
      )
    },
    assign_grnas = function(x) assign_grnas(x, parallel = FALSE),
    run_qc = function(x) run_qc(x, p_mito_threshold = P_MITO_THRESHOLD)
  )
  prepared <- TRUE
  for (name in names(preparation)) {
    out <- run_step(name, function() preparation[[name]](so))
    steps[[name]] <- out$record
    if (!is.null(out$record$error)) {
      prepared <- FALSE
      break
    }
    so <- out$value
  }

  results <- list()
  if (prepared) {
    analysis_fns <- list(
      run_calibration_check = function(x) run_calibration_check(x, parallel = FALSE, output_amount = OUTPUT_AMOUNT),
      run_power_check = function(x) run_power_check(x, parallel = FALSE, output_amount = OUTPUT_AMOUNT),
      run_discovery_analysis = function(x) run_discovery_analysis(x, parallel = FALSE, output_amount = OUTPUT_AMOUNT)
    )
    for (analysis in ANALYSES) {
      out <- run_step(analysis, function() analysis_fns[[analysis]](so))
      steps[[analysis]] <- out$record
      if (!is.null(out$record$error)) next
      so <- out$value
      df <- get_result(so, analysis)
      file <- paste0(analysis, ".csv")
      write_result_csv(df, file.path(run_dir, file))
      results[[analysis]] <- result_summary(df, file)
      cat(sprintf("  %s: %d rows, %s pass_qc, %s significant\n", analysis, results[[analysis]]$rows,
                  format(results[[analysis]]$pass_qc), format(results[[analysis]]$significant)))
    }
    cat("\n")
    try(print(so))
    cat("\n")
  }

  if (!is.null(so)) write_atomic(file.path(run_dir, "sceptre_object.rds"), function(tmp) saveRDS(so, tmp))

  errors <- Filter(Negate(is.null), lapply(steps, `[[`, "error"))
  status <- if (!prepared) "error" else if (length(errors)) "partial" else "ok"
  flush(log_con)
  log_lines <- gsub("\033\\[[0-9;]*m", "", readLines(log_path, warn = FALSE))
  printed_notes <- unique(trimws(grep("Warning|Note:|Error", log_lines, value = TRUE)))
  printed_notes <- printed_notes[!grepl("^\\[|^ERROR in |^R warning \\[", printed_notes)]

  info <- list(
    status = status,
    dataset = dataset,
    run = run_name,
    description = spec$description,
    versions = versions(),
    seed = SEED,
    pipeline = c(spec$pipeline, list(
      set_analysis_parameters = sprintf(
        "set_analysis_parameters(sceptre_object, discovery_pairs, positive_control_pairs, side = '%s', control_group = '%s', resampling_mechanism = '%s')",
        spec$side, control_group, resampling_mechanism
      ),
      assign_grnas = "assign_grnas(sceptre_object, parallel = FALSE) [default method]",
      run_qc = sprintf("run_qc(sceptre_object, p_mito_threshold = %s) [other thresholds default]", P_MITO_THRESHOLD),
      analyses = sprintf(
        "run_calibration_check, run_power_check, run_discovery_analysis in that order, parallel = FALSE, output_amount = %d, other arguments default",
        OUTPUT_AMOUNT
      )
    )),
    settings = if (!is.null(so)) object_summary(so) else NULL,
    wall_time_s = lapply(steps, `[[`, "wall_time_s"),
    results = results,
    errors = errors,
    warnings = Filter(length, lapply(steps, `[[`, "warnings")),
    messages = Filter(length, lapply(steps, `[[`, "messages")),
    printed_notes = printed_notes
  )
  write_atomic(file.path(run_dir, "run_info.json"), function(tmp) {
    jsonlite::write_json(info, tmp, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null", null = "null")
  })
  stamp("done ", dataset, "/", run_name, ": ", status)
  status
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  usage <- "usage: Rscript scripts/run_sceptredata_examples.R <out_dir> [lowmoi|highmoi|all]"
  if (length(args) < 1L || length(args) > 2L) stop(usage)
  out_dir <- args[[1]]
  if (!startsWith(out_dir, "/")) out_dir <- file.path(getwd(), out_dir)
  which_datasets <- if (length(args) == 2L) args[[2]] else "all"
  if (!which_datasets %in% c(names(RUNS), "all")) stop(usage)
  refuse_unless_ignored(out_dir)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  out_dir <- normalizePath(out_dir)

  data_env <- load_sceptredata()
  datasets <- if (which_datasets == "all") names(RUNS) else which_datasets
  statuses <- character()
  for (dataset in datasets) {
    for (run in RUNS[[dataset]]) {
      key <- paste0(dataset, "/", run[["control_group"]], "_", run[["resampling_mechanism"]])
      statuses[[key]] <- tryCatch(
        run_one(dataset, run[["control_group"]], run[["resampling_mechanism"]], data_env, out_dir),
        error = function(e) {
          cat("run ", key, " failed outside a step: ", conditionMessage(e), "\n", sep = "")
          "error"
        }
      )
    }
  }
  cat("\nSummary:\n")
  for (key in names(statuses)) cat(sprintf("  %-40s %s\n", key, statuses[[key]]))
  if (any(statuses %in% c("error", "partial"))) quit(status = 1L)
}

main()
