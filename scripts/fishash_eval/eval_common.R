# Helpers shared by the fishash evaluation's R scripts (sourced, not run).
#
# The conventions follow scripts/run_sceptredata_examples.R: refuse to write outside a
# git-ignored directory, write every file atomically, and record package versions and SHAs
# next to every output.

`%||%` <- function(a, b) if (is.null(a)) b else a

stamp <- function(...) cat(sprintf("[%s] ", format(Sys.time(), "%Y-%m-%d %H:%M:%S")), ..., "\n", sep = "")

# Parquet and JSON outputs are not covered by a .gitignore pattern, so a directory inside the
# work tree must be ignored as a whole (test_data/ is).
refuse_unless_ignored <- function(out_dir) {
  git <- Sys.which("git")
  if (!nzchar(git)) return(invisible())
  anchor <- out_dir
  while (!dir.exists(anchor)) anchor <- dirname(anchor)
  in_tree <- suppressWarnings(system2(git, c("-C", shQuote(anchor), "rev-parse", "--is-inside-work-tree"),
                                      stdout = TRUE, stderr = FALSE))
  if (!identical(in_tree, "true")) return(invisible())
  probe <- file.path(out_dir, "probe", "meta.json")
  ignored <- system2(git, c("-C", shQuote(anchor), "check-ignore", "-q", shQuote(probe)), stdout = FALSE, stderr = FALSE)
  if (ignored != 0L) {
    stop("refusing to write to ", out_dir, ": it is inside a git work tree and not ignored")
  }
}

write_atomic <- function(path, writer) {
  tmp <- paste0(path, ".tmp")
  writer(tmp)
  if (!file.rename(tmp, path)) stop("could not move ", tmp, " to ", path)
}

write_json_atomic <- function(x, path) {
  write_atomic(path, function(tmp) {
    writeLines(jsonlite::toJSON(x, auto_unbox = TRUE, digits = I(17), na = "string", null = "null",
                                pretty = TRUE), tmp)
  })
}

pkg_info <- function(pkg) {
  d <- tryCatch(utils::packageDescription(pkg), warning = function(w) NULL)
  if (is.null(d) || !is.list(d)) return(list(version = NA, remote_sha = NA))
  list(version = d$Version, remote_sha = d$RemoteSha %||% d$GithubSHA1 %||% NA)
}

versions <- function(pkgs = c("fishash", "extraDistr", "Matrix", "SummarizedExperiment", "sparseMatrixStats")) {
  out <- lapply(pkgs, pkg_info)
  names(out) <- pkgs
  c(out, list(
    r = R.version.string,
    platform = R.version$platform,
    sizeof_longdouble = .Machine$sizeof.longdouble,
    blas = utils::sessionInfo()$BLAS,
    lapack = La_library()
  ))
}

sha256_file <- function(path) digest::digest(file = path, algo = "sha256")

# A sparse matrix as (guide, cell, value) triplets, 0-based, in column-major order: the layout
# scripts/sceptre_export_lib.R already uses for parquet, read on the Python side into scipy.
sparse_triplets <- function(m, logical = FALSE) {
  m <- as(as(m, "CsparseMatrix"), "TsparseMatrix")
  ord <- order(m@j, m@i)
  out <- data.frame(guide = as.integer(m@i[ord]), cell = as.integer(m@j[ord]))
  if (!logical) {
    x <- m@x[ord]
    if (any(x != round(x))) stop("non-integer values in a count matrix")
    out$value <- as.integer(x)
  } else {
    if (!all(m@x[ord])) stop("stored FALSE in a logical matrix")
  }
  out
}

write_triplets <- function(m, path, logical = FALSE) {
  write_atomic(path, function(tmp) arrow::write_parquet(sparse_triplets(m, logical), tmp))
}
