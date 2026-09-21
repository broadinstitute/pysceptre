#!/usr/bin/env Rscript
# Ground truth for poscounts size factors, from DESeq2 itself.
#
# `analytical_power/inputs.py::poscounts_size_factors` reproduces the per-cell factors that the
# validated `expression_mean` is built on. DESeq2 is where the "poscounts" estimator is defined, so
# it is the reference rather than a second transcription of the same arithmetic: comparing a port
# against my own re-implementation of what it ports would test nothing.
#
# DESeq2 is not a dependency of anything else here and is not installed in CI, which is why this
# writes a cached fixture. tests/validation/conftest.py regenerates it only if the file is missing.
# If you change the matrices below, DELETE the JSON or the tests keep validating against the stale
# one.
#
# DESeq2 densifies, which is exactly why pysceptre does not use it: on a real screen the dense
# matrix is tens of GB. At the fixture's size that does not matter.
#
# WHAT THIS FIXTURE SETTLES, AND WHAT IT DOES NOT. DESeq2 divides its factors by their geometric
# mean, so `sizeFactors()` comes back centred on 1. The implementation the analytical power
# estimator was validated against does not do that, and the difference is a single constant per
# dataset: 1.12 and 1.19 on the two non-trivial cases here. It is not cosmetic, because
# `expression_mean` divides by these factors, so a constant on them scales every gene's mean and
# therefore every power estimate.
#
# So the Python test compares the *centred* form against DESeq2, which validates every ratio in
# the vector, and the absolute scale is validated separately against real output from the
# implementation that produced the published numbers. Neither check alone is enough.
#
# Usage:
#   Rscript scripts/dump_poscounts_ground_truth.R tests/validation/poscounts_ground_truth.json

suppressMessages(library(DESeq2))

args <- commandArgs(trailingOnly = TRUE)
out_path <- if (length(args) >= 1) args[1] else "tests/validation/poscounts_ground_truth.json"

num_fmt <- function(v) paste(vapply(v, function(z) formatC(z, digits = 17, format = "g"),
                                    character(1)), collapse = ", ")

# The cases span what makes this estimator awkward: a gene that is zero in every cell (excluded
# from the geometric means), a very sparse cell (its median is taken over few nonzeros), a cell
# with a single nonzero (median of one value), and wildly different library sizes, which is the
# whole reason size factors exist.
make_case <- function(label, m) {
  storage.mode(m) <- "integer"
  dds <- DESeqDataSetFromMatrix(countData = m,
                                colData = data.frame(x = factor(rep("a", ncol(m)))),
                                design = ~1)
  dds <- estimateSizeFactors(dds, type = "poscounts")
  sf <- sizeFactors(dds)
  # The normalised mean this feeds: divide each column by its factor, then average over cells.
  normalised_mean <- rowMeans(sweep(m, 2, sf, "/"))
  list(label = label, m = m, sf = as.numeric(sf), mean = as.numeric(normalised_mean))
}

set.seed(20260921)
cases <- list()

m <- matrix(rpois(12 * 25, lambda = 6), nrow = 12)
m[, 4] <- 0L; m[1, 4] <- 5L       # a cell with a single nonzero
m[, 7] <- rpois(12, lambda = 200) # a cell with a much larger library
m[3, ] <- 0L                      # a gene that is zero everywhere: LAST, or the two lines
                                  # above put counts back into it
stopifnot(all(m[3, ] == 0L), sum(m[, 4] != 0L) == 1L)
cases[[length(cases) + 1]] <- make_case("mixed", m)

m2 <- matrix(rpois(6 * 10, lambda = 3), nrow = 6)
m2[m2 > 2] <- 0L                  # very sparse
m2[, 1] <- c(1L, 0L, 0L, 0L, 0L, 2L)
for (k in seq_len(ncol(m2))) if (all(m2[, k] == 0L)) m2[1, k] <- 1L
cases[[length(cases) + 1]] <- make_case("sparse", m2)

m3 <- matrix(rep(4L, 5 * 8), nrow = 5)   # every cell identical: factors must all be 1
cases[[length(cases) + 1]] <- make_case("uniform", m3)

blocks <- vapply(cases, function(c) {
  sprintf(paste0('    {\n      "label": "%s",\n      "n_genes": %d,\n      "n_cells": %d,\n',
                 '      "counts_rowmajor": [%s],\n      "size_factors": [%s],\n',
                 '      "normalised_mean": [%s]\n    }'),
          c$label, nrow(c$m), ncol(c$m),
          num_fmt(as.numeric(t(c$m))), num_fmt(c$sf), num_fmt(c$mean))
}, character(1))

writeLines(c("{",
             sprintf('  "deseq2_version": "%s",', as.character(utils::packageVersion("DESeq2"))),
             sprintf('  "r_version": "%s",', paste(R.version$major, R.version$minor, sep = ".")),
             '  "cases": [',
             paste(blocks, collapse = ",\n"),
             "  ]",
             "}"), out_path)
cat(sprintf("wrote %s (%d cases, DESeq2 %s)\n", out_path, length(cases),
            as.character(utils::packageVersion("DESeq2"))))
