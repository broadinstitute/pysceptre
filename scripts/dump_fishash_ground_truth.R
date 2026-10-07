#!/usr/bin/env Rscript
# Ground truth for the fishash port (pysceptre.assignment), from the installed fishash package.
#
#   Rscript scripts/dump_fishash_ground_truth.R <out.json>
#
# Writes one JSON fixture; tests/validation/conftest.py gzips it to fishash_ground_truth.json.gz
# (mtime 0, so regenerating gives the same bytes). Treat it like the other committed ground-truth
# fixtures: if you change this script, DELETE the .gz and regenerate it, or the tests keep
# validating against a stale fixture. The fixture records this script's md5 and a test compares
# it with the script on disk.
#
# Sections
#   provenance   fishash version and install SHA, R version, platform, BLAS, LAPACK,
#                sizeof(long double) (R's phyper accumulates its series in long double, which is
#                plain double on arm64), and the md5 of this script.
#   phyper       stats::phyper(q, m, n, k, lower.tail, log.p = TRUE) on hand-picked branch points
#                and a systematic sweep, both tails. fishash computes every p-value with it.
#   impute_unit  direct calls of fishash::impute_masked_counts, with its inner iteration count,
#                guide frequencies and cell sizes captured by trace().
#   cases        small count matrices, simulated with fishash::simulate_guidebender2 under a fixed
#                seed or built by hand, run through fishash::fishash() under several arguments.
#                Per-pass internals are captured with trace() on fishash_internal and
#                impute_masked_counts; every traced run is checked identical() to an untraced one.
#
# Synthetic data only. Conventions: indices are 0-based; a count matrix is column-major triplets
# {i, j, x} including any stored zeros; every per-entry vector follows that order; index sets are
# arrays even at length 1; non-finite numbers are the strings "Inf", "-Inf", "NaN", "NA".

suppressPackageStartupMessages({
  library(Matrix)
  library(SummarizedExperiment)
  library(fishash)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("usage: dump_fishash_ground_truth.R <out.json>")
out_path <- args[[1]]
script_path <- normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)))

`%||%` <- function(a, b) if (is.null(a)) b else a
arr <- function(x) I(x)
pkg <- function(p) {
  d <- utils::packageDescription(p)
  list(version = d$Version, remote_sha = d$RemoteSha %||% NA_character_)
}

# ---------------------------------------------------------------------------------------------
# Matrices as triplets
# ---------------------------------------------------------------------------------------------

triplets <- function(m) {
  m <- as(as(m, "CsparseMatrix"), "TsparseMatrix")
  o <- order(m@j, m@i)
  list(i = arr(as.integer(m@i[o])), j = arr(as.integer(m@j[o])), x = arr(as.numeric(m@x[o])))
}

entry_key <- function(i, j, nrow) as.numeric(j) * nrow + as.numeric(i)

# Positions (0-based, in the counts' column-major entry order) of the TRUE entries of a logical
# matrix whose TRUE entries all sit on stored entries of the counts.
true_positions <- function(lmat, counts_trip, nrow) {
  t <- as(as(lmat, "CsparseMatrix"), "TsparseMatrix")
  keep <- t@x
  keys <- entry_key(t@i[keep], t@j[keep], nrow)
  ckeys <- entry_key(counts_trip$i, counts_trip$j, nrow)
  pos <- match(keys, ckeys)
  if (anyNA(pos)) stop("an assigned entry is not a stored count")
  arr(as.integer(sort(pos) - 1L))
}

# A per-entry data frame from fishash_internal (any row order) back in the counts' entry order.
in_entry_order <- function(df, counts_trip, nrow) {
  keys <- entry_key(df$row_idx - 1, df$col_idx - 1, nrow)
  ckeys <- entry_key(counts_trip$i, counts_trip$j, nrow)
  df[match(ckeys, keys), , drop = FALSE]
}

# ---------------------------------------------------------------------------------------------
# Recording fishash's internals
# ---------------------------------------------------------------------------------------------

rec <- new.env()
reset_rec <- function() {
  rec$passes <- list()
  rec$imputes <- list()
}

record_pass <- function(env) {
  df <- env$df
  rec$passes[[length(rec$passes) + 1L]] <- list(
    df = df[, c("row_idx", "col_idx", "count", "row_sum", "tot", "log_pval", "odds_ratio",
                "odds_ratio_regularized")],
    background = env$background,
    cutoff = env$logpval_cutoff,
    B = if (exists("B", envir = env, inherits = FALSE)) get("B", envir = env) else NA,
    n_signif = if (exists("n_signif", envir = env, inherits = FALSE)) get("n_signif", envir = env) else NA,
    assigned = env$mat_assigned
  )
}

record_impute <- function(env) {
  rec$imputes[[length(rec$imputes) + 1L]] <- list(
    n_iter = env$i, mask = env$mask, guide_freqs = env$guide_freqs, cell_sizes = env$cell_sizes,
    out = env$counts
  )
}

ns <- asNamespace("fishash")
trace_on <- function() {
  suppressMessages({
    trace("fishash_internal", where = ns, print = FALSE, exit = bquote(.(record_pass)(environment())))
    trace("impute_masked_counts", where = ns, print = FALSE, exit = bquote(.(record_impute)(environment())))
  })
}
trace_off <- function() {
  suppressMessages({
    untrace("fishash_internal", where = ns)
    untrace("impute_masked_counts", where = ns)
  })
}

num <- function(x) {
  x <- as.numeric(x)
  out <- as.list(x)
  out[is.na(x) & !is.nan(x)] <- "NA"
  out[is.nan(x)] <- "NaN"
  out[is.infinite(x) & x > 0] <- "Inf"
  out[is.infinite(x) & x < 0] <- "-Inf"
  out
}
scalar <- function(x) if (length(x) == 1 && is.finite(x)) as.numeric(x) else num(x)[[1]]

# Runs fishash() twice, untraced and traced, and returns the record of one argument set.
run_case <- function(counts, args, detail = c("summary", "deep"), odds = FALSE) {
  detail <- match.arg(detail)
  counts_trip <- triplets(counts)
  nrow <- nrow(counts)
  plain <- tryCatch(do.call(fishash::fishash, c(list(counts), args)), error = function(e) e)
  record <- list(args = args[setdiff(names(args), "background")],
                 background = !is.null(args$background), error = NULL)
  if (inherits(plain, "error")) {
    record$error <- conditionMessage(plain)
    return(record)
  }
  reset_rec()
  trace_on()
  traced <- tryCatch(do.call(fishash::fishash, c(list(counts), args)), finally = trace_off())
  stopifnot(identical(traced, plain))

  res <- plain
  passes <- rec$passes
  last <- passes[[length(passes)]]
  last_df <- in_entry_order(last$df, counts_trip, nrow)
  record$num_iter <- metadata(res)$num_iter
  record$cutoff <- scalar(metadata(res)$log_pval_cutoff)
  record$assigned <- true_positions(assay(res, "assigned"), counts_trip, nrow)
  record$demux_type <- arr(as.character(colData(res)$demux_type))
  record$assignment <- arr(as.character(colData(res)$assignment))
  record$log_pval <- num(last_df$log_pval)
  if (odds || detail == "deep") {
    record$odds_ratio <- num(last_df$odds_ratio)
    record$odds_ratio_regularized <- num(last_df$odds_ratio_regularized)
  }
  margin <- abs(last_df$log_pval - last$cutoff)
  record$min_abs_margin <- scalar(min(margin[is.finite(margin)], Inf))
  record$passes <- lapply(seq_along(passes), function(p) {
    ps <- passes[[p]]
    df <- in_entry_order(ps$df, counts_trip, nrow)
    out <- list(
      cutoff = scalar(ps$cutoff),
      B = scalar(ps$B),
      n_signif = scalar(ps$n_signif),
      assigned = true_positions(ps$assigned, counts_trip, nrow),
      min_abs_margin = scalar({
        mg <- abs(df$log_pval - ps$cutoff)
        min(mg[is.finite(mg)], Inf)
      }),
      min_half_distance = scalar({
        # distance of each phyper margin from .5, where R's half-to-even rounding would bite
        fr <- c(df$row_sum, df$tot - df$row_sum)
        fr <- abs((fr - floor(fr)) - 0.5)
        min(fr[is.finite(fr)], Inf)
      })
    )
    if (detail == "deep") {
      # The background at the counts' entries is what the next pass needs; row_sum and tot follow
      # from it, so the stepwise test feeds it in and checks the log p-values.
      bg <- ps$background
      out$background <- num(as.numeric(bg[cbind(df$row_idx, df$col_idx)]))
      out$log_pval <- num(df$log_pval)
    }
    out
  })
  record$imputes <- lapply(rec$imputes, function(im) {
    out <- list(n_iter = as.integer(im$n_iter))
    if (detail == "deep") {
      mpos <- true_positions(im$mask != 0, counts_trip, nrow)
      out$mask <- mpos
      out$guide_freqs <- num(im$guide_freqs)
      out$cell_sizes <- num(im$cell_sizes)
      o <- as.numeric(im$out[cbind(counts_trip$i[mpos + 1] + 1, counts_trip$j[mpos + 1] + 1)])
      out$imputed <- num(o)
    }
    out
  })
  record
}

case_record <- function(id, counts, runs, seed = NA, note = "") {
  stopifnot(!is.null(rownames(counts)))
  list(
    id = id, seed = seed, note = note, dims = arr(dim(counts)),
    grna_ids = arr(rownames(counts)),
    counts = triplets(counts),
    runs = runs
  )
}

# ---------------------------------------------------------------------------------------------
# phyper grid
# ---------------------------------------------------------------------------------------------

phyper_grid <- function() {
  hand <- list(
    # support edges and the swap boundary
    c(-1, 5, 5, 3), c(0, 5, 5, 3), c(2, 5, 5, 3), c(3, 5, 5, 3), c(4, 5, 5, 3),
    c(1, 3, 7, 6), c(2, 3, 7, 6), c(5, 10, 10, 10), c(4, 10, 10, 8), c(0, 0, 10, 4),
    c(3, 10, 0, 3), c(9, 10, 0, 10), c(0, 1, 1, 1), c(0, 1, 1, 2), c(1, 1, 1, 2),
    # domain errors and NaN
    c(1, -1, 5, 2), c(1, 5, -1, 2), c(1, 5, 5, 11), c(1, 5, 5, -1), c(NaN, 5, 5, 2),
    # q floored after + 1e-7, margins at .5 (half to even) and other fractions
    c(2.9999999, 50, 50, 20), c(2.5, 50, 50, 20), c(3, 2.5, 7.5, 5), c(3, 3.5, 6.5, 5),
    c(3, 4.5, 5.5, 5), c(3, 5.5, 4.5, 5), c(3, 2.49, 7.51, 5), c(3, 2.51, 7.49, 5),
    # x = 0 with p above and below q, and x == k; dbinom_raw's x == 0 branch with p > q needs
    # more than half the urn drawn (0, 10, 100, 60), and its x == n branch with p > q needs
    # x = k - n (5, 10, 20, 25)
    c(0, 50, 50, 30), c(0, 50, 50, 80), c(29, 50, 50, 30), c(30, 50, 50, 30),
    c(0, 10, 100, 60), c(0, 20, 200, 150), c(5, 10, 20, 25), c(12, 30, 60, 72),
    # deep tails and long pdhyper series
    c(400, 500, 1999500, 500), c(999, 1000, 1999000, 1000), c(50, 200, 1e6, 100),
    c(1500, 3000, 3000, 3000), c(2000, 5e5, 5e5, 4000), c(1100, 1e4, 1e4, 2000),
    # stirlerr intervals: 16..23, 24..27, 28..86, 87..205, 206..6180, > 6180, > 15.7e6
    c(10, 20, 20, 23), c(12, 25, 25, 26), c(30, 60, 60, 70), c(90, 150, 150, 180),
    c(500, 3000, 3000, 1000), c(4000, 8000, 8000, 8000), c(3, 10, 2e7, 50), c(5000, 9e6, 9e6, 1e4),
    # bd0's Taylor branch (near the mean) and its log branch
    c(50, 1000, 1000, 100), c(51, 1000, 1000, 100), c(5, 1000, 1000, 100), c(95, 1000, 1000, 100)
  )
  hand <- do.call(rbind, hand)
  colnames(hand) <- c("q", "m", "n", "k")
  sweep <- expand.grid(
    x = c(0, 1, 2, 3, 5, 10, 30, 100, 1000),
    k = c(1, 2, 5, 20, 100, 1000, 5000),
    m = c(1, 5, 50, 500, 5e3, 5e4, 5e5),
    tot = c(2e3, 2e5, 2e6, 2e7)
  )
  sweep <- sweep[sweep$x <= sweep$k & sweep$k <= sweep$tot & sweep$m <= sweep$tot, ]
  shifted <- lapply(c(0, 0.37, 0.5), function(s) {
    data.frame(q = sweep$x - 1, m = sweep$m + s, n = sweep$tot - sweep$m + s, k = sweep$k)
  })
  grid <- rbind(as.data.frame(hand), do.call(rbind, shifted))
  grid$hand_picked <- c(rep(TRUE, nrow(hand)), rep(FALSE, nrow(grid) - nrow(hand)))
  upper <- suppressWarnings(phyper(grid$q, grid$m, grid$n, grid$k, lower.tail = FALSE, log.p = TRUE))
  lower <- suppressWarnings(phyper(grid$q, grid$m, grid$n, grid$k, lower.tail = TRUE, log.p = TRUE))
  list(
    q = num(grid$q), m = num(grid$m), n = num(grid$n), k = num(grid$k),
    hand_picked = arr(grid$hand_picked), upper = num(upper), lower = num(lower)
  )
}

# ---------------------------------------------------------------------------------------------
# impute_masked_counts, called directly
# ---------------------------------------------------------------------------------------------

impute_record <- function(label, counts, mask, eps = 1e-4, max_iter = 10) {
  reset_rec()
  trace_on()
  out <- tryCatch(fishash::impute_masked_counts(counts, mask, eps = eps, max_iter = max_iter),
                  finally = trace_off())
  plain <- fishash::impute_masked_counts(counts, mask, eps = eps, max_iter = max_iter)
  stopifnot(identical(out, plain))
  im <- rec$imputes[[1]]
  list(
    label = label, dims = arr(dim(counts)), eps = eps, max_iter = max_iter,
    counts = triplets(counts), mask = triplets(as(mask != 0, "CsparseMatrix") * 1),
    out = triplets(out), n_iter = as.integer(im$n_iter),
    guide_freqs = num(im$guide_freqs), cell_sizes = num(im$cell_sizes)
  )
}

impute_units <- function() {
  doc_counts <- Matrix::Matrix(c(1, 0, 2, 3, 0, 1, 0, 4, 2), nrow = 3, sparse = TRUE)
  doc_mask <- Matrix::Matrix(c(0, 1, 0, 0, 0, 1, 0, 0, 1), nrow = 3, sparse = TRUE)
  set.seed(7)
  m12 <- Matrix::rsparsematrix(12, 20, density = 0.6, rand.x = function(n) rpois(n, 6) + 1)
  m12 <- as(round(m12), "CsparseMatrix")
  nz <- which(as.matrix(m12) != 0)
  maskv <- rep(0, length(m12))
  maskv[sample(nz, round(0.15 * length(nz)))] <- 1
  mask12 <- Matrix::Matrix(maskv, nrow = 12, sparse = TRUE)
  list(
    impute_record("doc_example", doc_counts, doc_mask),
    impute_record("random_12x20", m12, mask12),
    impute_record("random_12x20_all_iterations", m12, mask12, eps = 1e-15)
  )
}

# ---------------------------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------------------------

simulated <- function(seed, ...) {
  set.seed(seed)
  sim <- fishash::simulate_guidebender2(return_sparse_only = TRUE, ...)
  assay(sim, "counts")
}

margins_ok <- function(rec_run, need_deep_iters = 0) {
  if (!is.null(rec_run$error)) return(FALSE)
  ok_margin <- all(vapply(rec_run$passes, function(p) {
    is.character(p$min_abs_margin) || p$min_abs_margin >= 1e-6
  }, logical(1)))
  ok_half <- all(vapply(rec_run$passes, function(p) {
    is.character(p$min_half_distance) || p$min_half_distance >= 1e-6
  }, logical(1)))
  ok_margin && ok_half && rec_run$num_iter >= need_deep_iters
}

GS <- function(...) list(padj_cutoff = 0.05, padj_method = "GS", ...)

# Simulated cases: the first seed in a fixed list whose runs keep every margin away from the cut
# and from .5, so a last-bit difference cannot flip a call. The deep run must reach pass 5, so
# the "mask OR assigned" rule after pass 3 is exercised.
seed_list <- c(1:60)
find_sim <- function(id, sim_args, run_specs, deep_index = NA, need_iters = 0, note = "",
                     odds_index = integer()) {
  for (s in seed_list) {
    counts <- do.call(simulated, c(list(seed = s), sim_args))
    rownames(counts) <- paste0("g", seq_len(nrow(counts)))
    colnames(counts) <- paste0("c", seq_len(ncol(counts)))
    runs <- lapply(seq_along(run_specs), function(r) {
      run_case(counts, run_specs[[r]], detail = if (!is.na(deep_index) && r == deep_index) "deep" else "summary",
               odds = r %in% odds_index)
    })
    ok <- all(vapply(seq_along(runs), function(r) {
      if (!is.null(runs[[r]]$error)) return(FALSE)
      margins_ok(runs[[r]], need_deep_iters = if (!is.na(deep_index) && r == deep_index) need_iters else 0)
    }, logical(1)))
    if (ok) return(case_record(id, counts, runs, seed = s, note = note))
  }
  stop("no seed in the list satisfies the invariants for case ", id)
}

hand_case <- function(id, counts, run_specs, note = "", check_margins = TRUE) {
  rownames(counts) <- paste0("g", seq_len(nrow(counts)))
  colnames(counts) <- paste0("c", seq_len(ncol(counts)))
  runs <- lapply(run_specs, function(a) run_case(counts, a))
  if (check_margins) {
    for (r in runs) if (is.null(r$error) && !margins_ok(r)) stop("case ", id, " has a margin within 1e-6")
  }
  case_record(id, counts, runs, note = note)
}

build_cases <- function() {
  cases <- list()
  sim_a_args <- list(n_guides = 30, n_cells = 200, moi = 0.3, hurdle_prob = 0.1, snr = 4,
                     count_per_cell = 50, frac_noise_endo = 0.75)
  cases$sim_a <- find_sim(
    "sim_a", sim_a_args,
    list(GS(refit = 10), GS(refit = 0), GS(refit = 1),
         list(padj_cutoff = 0.05, padj_method = "BH", refit = 10),
         list(padj_cutoff = 0.05, padj_method = "BY", refit = 0),
         GS(refit = 10, exclude_empty = FALSE)),
    deep_index = 1, need_iters = 5, odds_index = 2L,
    note = "simulate_guidebender2, 30 guides x 200 cells, moi 0.3; run 1 is the deep run"
  )
  cases$sim_b <- find_sim(
    "sim_b", modifyList(sim_a_args, list(moi = 3)),
    list(GS(refit = 10), GS(refit = 0)), note = "moi 3: many guides per cell"
  )
  cases$sim_c <- find_sim(
    "sim_c", modifyList(sim_a_args, list(snr = 1, count_per_cell = 20, frac_noise_endo = 0.25,
                                         endo_shape_flat = 1)),
    list(GS(refit = 0), GS(refit = 10)),
    note = "low signal, flat endogenous noise: signal and noise guide frequencies uncorrelated"
  )
  cases$sim_d <- find_sim(
    "sim_d", modifyList(sim_a_args, list(Phi_noise = 1)),
    list(GS(refit = 10)), note = "overdispersed noise (Phi_noise = 1)"
  )

  # Two empty guides (rows 3 and 7) and three empty cells (columns 2, 9, 14).
  set.seed(101)
  e <- matrix(rpois(10 * 14, 1.2), 10, 14)
  e[c(3, 7), ] <- 0
  e[, c(2, 9, 14)] <- 0
  e[1, 1] <- 12
  e[2, 4] <- 15
  e[5, 6] <- 9
  cases$empty <- hand_case("empty", Matrix::Matrix(e, sparse = TRUE),
    list(GS(refit = 10, exclude_empty = TRUE), GS(refit = 10, exclude_empty = FALSE),
         list(padj_cutoff = 0.05, padj_method = "BH", refit = 0, exclude_empty = FALSE)),
    note = "empty guides and cells change the number of tests only under exclude_empty")

  set.seed(102)
  ec <- matrix(rpois(12 * 40, 0.6), 12, 40)
  ec[1, ] <- ec[1, ] + 1 + rpois(40, 3)
  ec[cbind(sample(2:12, 40, TRUE), 1:40)] <- 10 + rpois(40, 10)
  cases$every_cell <- hand_case("every_cell", Matrix::Matrix(ec, sparse = TRUE),
    list(GS(refit = 10)), note = "guide 1 is nonzero in every cell")

  # Guide 2 appears only where it is assigned; cell 1's counts are all assigned; cell 3 holds one
  # guide only. After pass 1 the unmasked counts of guide 2 and of cell 1 are all zero (ff4de6b).
  set.seed(103)
  uz <- matrix(rpois(15 * 60, 0.5), 15, 60)
  uz[2, ] <- 0
  uz[2, c(5, 17, 33)] <- c(40, 35, 50)
  uz[, 1] <- 0
  uz[c(4, 9), 1] <- c(30, 45)
  uz[, 3] <- 0
  uz[6, 3] <- 25
  for (j in 4:60) uz[sample(c(1, 3:15), 1), j] <- uz[sample(c(1, 3:15), 1), j] + 20
  cases$unmasked_zero <- hand_case("unmasked_zero", Matrix::Matrix(uz, sparse = TRUE),
    list(GS(refit = 10)), note = "rows and columns whose unmasked counts all vanish after pass 1",
    check_margins = FALSE)

  ad <- matrix(0, 6, 12)
  for (g in 1:6) ad[g, (2 * g - 1):(2 * g)] <- 50
  cases$all_assigned <- hand_case("all_assigned", Matrix::Matrix(ad, sparse = TRUE),
    list(GS(refit = 10), GS(refit = 0)), note = "block diagonal: every nonzero entry is assigned",
    check_margins = FALSE)

  set.seed(104)
  ti <- matrix(rpois(8 * 30, 1), 8, 30)
  ti[cbind(sample(1:8, 30, TRUE), 1:30)] <- 12
  ti <- cbind(ti, ti[, 1:6])
  ti <- rbind(ti, ti[2, ])
  cases$ties <- hand_case("ties", Matrix::Matrix(ti, sparse = TRUE),
    list(GS(refit = 10), list(padj_cutoff = 0.05, padj_method = "BH", refit = 10),
         list(padj_cutoff = 0.05, padj_method = "BY", refit = 10)),
    note = "duplicated cells and a duplicated guide: ties in every ordering", check_margins = FALSE)

  set.seed(105)
  ns_ <- matrix(rpois(20 * 50, 0.4), 20, 50)
  cases$no_signal <- hand_case("no_signal", Matrix::Matrix(ns_, sparse = TRUE),
    list(GS(refit = 0), list(padj_cutoff = 0.05, padj_method = "BH", refit = 0)),
    note = "noise only: GS finds B = 0 and BH finds n_signif = 0", check_margins = FALSE)

  mf <- matrix(0, 6, 8)
  mf[, 1] <- c(3, 7, 0, 0, 0, 0)
  mf[, 2] <- c(9, 21, 0, 0, 0, 0)
  mf[, 3] <- c(2, 5, 0, 0, 0, 0)
  mf[, 4] <- c(1, 2, 0, 0, 0, 0)
  mf[, 5] <- c(0, 0, 3, 6, 0, 0)
  mf[, 6] <- c(0, 0, 0, 0, 30, 70)
  mf[, 7] <- c(0, 0, 0, 0, 33, 77)
  mf[, 8] <- c(1, 1, 1, 1, 1, 5)
  cases$min_frac <- hand_case("min_frac", Matrix::Matrix(mf, sparse = TRUE),
    list(GS(refit = 0, min_frac = 0.3, min_count = 0), GS(refit = 0, min_frac = 0.3),
         GS(refit = 0, min_count = 0), GS(refit = 0, min_count = 5),
         list(padj_cutoff = 0.5, padj_method = "BH", refit = 0, min_frac = 0.3, min_count = 0)),
    note = "fractions near 0.3 computed as count * (1 / cell total)", check_margins = FALSE)

  bg_counts <- cases$sim_a$counts
  sim_a_mat <- Matrix::sparseMatrix(i = unclass(bg_counts$i) + 1L, j = unclass(bg_counts$j) + 1L,
                                    x = unclass(bg_counts$x), dims = unclass(cases$sim_a$dims))
  set.seed(106)
  bgm <- sim_a_mat * 0.7 + Matrix::rsparsematrix(nrow(sim_a_mat), ncol(sim_a_mat), 0.3,
                                                 rand.x = function(n) rgamma(n, 1, 2))
  bgm@x <- abs(bgm@x)
  rownames(sim_a_mat) <- paste0("g", seq_len(nrow(sim_a_mat)))
  colnames(sim_a_mat) <- paste0("c", seq_len(ncol(sim_a_mat)))
  cases$background <- case_record("background", sim_a_mat, list(
    run_case(sim_a_mat, GS(refit = 0, background = bgm), odds = TRUE),
    run_case(sim_a_mat, GS(refit = 1, background = bgm))
  ), note = "a non-integer background with a different pattern; refit > 0 must error")
  cases$background$background <- triplets(bgm)

  half_bg <- sim_a_mat
  set.seed(107)
  hx <- sample(c(0, 0.5, 1.5), length(half_bg@x), TRUE)
  half_bg@x <- half_bg@x + hx
  cases$half <- case_record("half", sim_a_mat, list(
    run_case(sim_a_mat, GS(refit = 0, background = half_bg))
  ), note = "background in multiples of 0.5: phyper's margins sit on .5, rounded half to even")
  cases$half$background <- triplets(half_bg)

  og <- Matrix::Matrix(matrix(c(3, 0, 5, 1, 9, 2), 1, 6), sparse = TRUE)
  oc <- Matrix::Matrix(matrix(c(3, 0, 5, 1, 9, 2), 6, 1), sparse = TRUE)
  set.seed(108)
  tc <- Matrix::Matrix(matrix(rpois(10, 3), 5, 2), sparse = TRUE)
  cases$one_guide <- hand_case("one_guide", og, list(GS(refit = 0), list(padj_cutoff = 0.05, padj_method = "BH", refit = 0)), check_margins = FALSE)
  cases$one_cell <- hand_case("one_cell", oc, list(GS(refit = 0), list(padj_cutoff = 0.05, padj_method = "BH", refit = 0)), check_margins = FALSE)
  cases$two_cells <- hand_case("two_cells", tc, list(GS(refit = 10), list(padj_cutoff = 0.05, padj_method = "BH", refit = 0)), check_margins = FALSE)

  ez <- sim_a_mat
  set.seed(109)
  zero_at <- sample(which(as.matrix(ez) == 0), 30)
  ez_t <- as(ez, "TsparseMatrix")
  zi <- (zero_at - 1) %% nrow(ez)
  zj <- (zero_at - 1) %/% nrow(ez)
  ez_t <- new("dgTMatrix", i = c(ez_t@i, as.integer(zi)), j = c(ez_t@j, as.integer(zj)),
              x = c(ez_t@x, rep(0, 30)), Dim = dim(ez), Dimnames = dimnames(ez))
  ez_c <- as(ez_t, "CsparseMatrix")
  stopifnot(sum(ez_c@x == 0) == 30)
  cases$explicit_zeros <- case_record("explicit_zeros", ez_c, list(run_case(ez_c, GS(refit = 10))),
                                      note = "sim_a with 30 stored zeros, which test as log p = 0")
  cases
}

# ---------------------------------------------------------------------------------------------

out <- list(
  provenance = list(
    fishash = pkg("fishash"),
    extraDistr = pkg("extraDistr"),
    Matrix = pkg("Matrix"),
    sparseMatrixStats = pkg("sparseMatrixStats"),
    jsonlite = pkg("jsonlite"),
    r_version = R.version.string,
    platform = R.version$platform,
    sizeof_longdouble = .Machine$sizeof.longdouble,
    blas = utils::sessionInfo()$BLAS,
    lapack = La_library(),
    dumper_md5 = unname(tools::md5sum(script_path))
  ),
  phyper = phyper_grid(),
  impute_unit = impute_units(),
  cases = unname(build_cases())
)

json <- jsonlite::toJSON(out, auto_unbox = TRUE, digits = I(17), na = "string", null = "null")
writeLines(json, out_path)
cat("wrote", out_path, format(file.size(out_path), big.mark = ","), "bytes\n")
