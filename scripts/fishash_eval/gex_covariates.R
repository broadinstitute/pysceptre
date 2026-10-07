#!/usr/bin/env Rscript
# Regenerate the gene expression the fishash paper gave sceptre, and record sceptre's covariates.
#
# Usage:
#   Rscript scripts/fishash_eval/gex_covariates.R <sims_dir> <scenario> [sim_label ...]
#
# sceptre's mixture method fits a Poisson GLM on cell covariates that include the gene
# expression's, and the simulations have no gene expression. The paper made some up per
# dataset (bin/get_gex_matrix.R): splatter's default simulation, one cell per simulated cell,
# seeded from "create_sceptre_obj.R_<in_rds>". Nextflow staged the Rds under its own basename
# and passed that as --in_rds, so the seed string is "create_sceptre_obj.R_<sim_label>.Rds".
# The sceptre object is then built as bin/run_sceptre_mixture.R builds it.
#
# Outputs, under <sims_dir>/<scenario>/<sim_label>/:
#   covariates.parquet   sceptre_object@covariate_data_frame as is, one row per cell, after a
#                        0-based "cell" column
#   covariates.json      the seed, the gex's dimensions and sha256, the default mixture formula
#                        sceptre derives from these covariates, and package versions
# The gex matrix is not saved. run_r_methods.R sources this file, regenerates the gex with the
# functions below and checks it against both outputs. Without sim_labels every dataset in the
# scenario's manifest is done; a dataset whose covariates.json exists is skipped.

suppressPackageStartupMessages({
  library(Matrix)
  library(SummarizedExperiment)
  library(SingleCellExperiment)
  library(sceptre)
})

gex_seed_string <- function(rds_name) paste0("create_sceptre_obj.R", "_", rds_name)

gex_seed <- function(rds_name) abs(digest::digest2int(gex_seed_string(rds_name)))

# bin/get_gex_matrix.R reads a "gex" attribute instead when the simulation carries one.
load_guide_counts <- function(rds_path) {
  se <- readRDS(rds_path)
  if (!is.null(attr(se, "gex"))) stop(rds_path, " carries a gex attribute; the paper used it instead of splatter")
  assay(se, "counts")
}

# bin/get_gex_matrix.R's splatter branch: a dense integer matrix, genes x cells.
regenerate_gex <- function(guide_counts, rds_name) {
  set.seed(gex_seed(rds_name))
  sce <- suppressMessages(splatter::splatSimulate(batchCells = ncol(guide_counts)))
  colnames(sce) <- colnames(guide_counts)
  counts(sce)
}

gex_sha256 <- function(gex) digest::digest(gex, algo = "sha256")

# bin/run_sceptre_mixture.R's import_data() call.
import_like_upstream <- function(gex, guide_counts) {
  import_data(
    response_matrix = gex,
    grna_matrix = guide_counts,
    grna_target_data_frame = data.frame(
      grna_id = rownames(guide_counts),
      grna_target = "non-targeting"
    ),
    moi = "high"
  )
}

# The formula assign_grnas(method = "mixture") uses when none is given.
default_mixture_formula <- function(covariate_data_frame) {
  sceptre:::auto_construct_formula_object(covariate_data_frame, include_grna_covariates = TRUE)
}

formula_text <- function(f) deparse1(f, collapse = " ")

read_manifest <- function(scenario_dir) {
  manifest <- jsonlite::read_json(file.path(scenario_dir, "manifest.json"))
  if (!identical(manifest$status, "ok")) stop("manifest of ", scenario_dir, " is not complete")
  manifest
}

select_datasets <- function(manifest, sim_labels) {
  if (length(sim_labels) == 0L) return(manifest$datasets)
  known <- vapply(manifest$datasets, function(d) d$sim_label, character(1))
  unknown <- setdiff(sim_labels, known)
  if (length(unknown) > 0L) stop("not in the manifest: ", paste(unknown, collapse = ", "))
  manifest$datasets[match(sim_labels, known)]
}

main <- function() {
  script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
  source(file.path(script_dir, "eval_common.R"))

  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) < 2L) stop("usage: gex_covariates.R <sims_dir> <scenario> [sim_label ...]")
  sims_dir <- args[[1]]
  scenario <- args[[2]]
  scenario_dir <- file.path(sims_dir, scenario)
  manifest <- read_manifest(scenario_dir)
  datasets <- select_datasets(manifest, args[-(1:2)])
  refuse_unless_ignored(scenario_dir)

  for (d in datasets) {
    out_dir <- file.path(scenario_dir, d$sim_label)
    json_path <- file.path(out_dir, "covariates.json")
    if (file.exists(json_path)) {
      stamp("skip ", d$sim_label, ": covariates.json exists")
      next
    }
    if (!dir.exists(out_dir)) stop(out_dir, " does not exist; run simulate.R first")

    guide_counts <- load_guide_counts(file.path(scenario_dir, "rds", d$rds))
    gex_seconds <- system.time(gex <- regenerate_gex(guide_counts, d$rds))[["elapsed"]]
    invisible(gc())
    gex_info <- list(
      class = class(gex)[[1]],
      type = typeof(gex),
      n_genes = nrow(gex),
      n_cells = ncol(gex),
      sha256 = gex_sha256(gex)
    )
    import_seconds <- system.time(sceptre_object <- import_like_upstream(gex, guide_counts))[["elapsed"]]
    rm(gex)
    invisible(gc())

    covariates <- sceptre_object@covariate_data_frame
    if (nrow(covariates) != ncol(guide_counts)) {
      stop("covariates have ", nrow(covariates), " rows for ", ncol(guide_counts), " cells")
    }
    if ("cell" %in% names(covariates)) stop("sceptre's covariates already have a 'cell' column")
    out <- data.frame(cell = seq_len(nrow(covariates)) - 1L, covariates, check.names = FALSE)
    rownames(out) <- NULL
    write_atomic(file.path(out_dir, "covariates.parquet"), function(tmp) arrow::write_parquet(out, tmp))

    write_json_atomic(list(
      scenario = scenario,
      sim_label = d$sim_label,
      rds = d$rds,
      seed_string = gex_seed_string(d$rds),
      seed = gex_seed(d$rds),
      rng_kind = RNGkind(),
      splatter_version = as.character(utils::packageVersion("splatter")),
      gex = gex_info,
      default_mixture_formula = formula_text(default_mixture_formula(covariates)),
      columns = names(covariates),
      column_types = lapply(covariates, typeof),
      n_cells = nrow(covariates),
      gex_seconds = gex_seconds,
      import_seconds = import_seconds,
      upstream_sha = manifest$upstream_sha,
      upstream_scripts = c("bin/get_gex_matrix.R", "bin/run_sceptre_mixture.R"),
      versions = versions(c("sceptre", "splatter", "fishash", "Matrix"))
    ), json_path)
    stamp("covariates for ", d$sim_label, " (gex ", gex_info$n_genes, " x ", gex_info$n_cells,
          ", ", round(gex_seconds, 1), " s)")
  }
  stamp("done ", scenario, ": ", length(datasets), " datasets")
}

if (sys.nframe() == 0L) main()
