#!/usr/bin/env Rscript
# Ground truth for the singleton pair expansion, from sceptre's own function.
#
# `grna_integration_strategy = "singleton"` does not change the statistical test at all. It changes
# which cells count as treated, by changing what sceptre calls a `grna_group`:
# `update_dfs_based_on_grouping_strategy` sets `grna_group = grna_id` instead of `grna_target`, and
# fans every (response, target) pair out to one row per guide of that target with a many-to-many
# join. Everything downstream is the union path unchanged.
#
# That join is the piece worth pinning against their code rather than transcribing, because the
# gRNA-to-target map is many-to-many in real data: a guide inside two overlapping candidate
# elements belongs to both targets, and the expansion has to keep it under each.
#
# The cases below are run through `sceptre:::update_dfs_based_on_grouping_strategy` on a minimal
# sceptre_object, which is the real function rather than a reading of it. tests/validation/
# conftest.py regenerates this only if the JSON is missing; delete it if you change the cases.
#
# Usage:
#   Rscript scripts/dump_singleton_ground_truth.R tests/validation/singleton_ground_truth.json

suppressPackageStartupMessages(library(sceptre))

args <- commandArgs(trailingOnly = TRUE)
out_path <- if (length(args) >= 1) args[1] else "tests/validation/singleton_ground_truth.json"

expand <- function(gtdf, pairs) {
  so <- methods::new("sceptre_object")
  so@grna_integration_strategy <- "singleton"
  so@grna_target_data_frame <- gtdf
  so@discovery_pairs <- pairs
  so@positive_control_pairs <- data.frame(grna_target = character(0), response_id = character(0))
  out <- sceptre:::update_dfs_based_on_grouping_strategy(so)
  list(pairs = out@discovery_pairs, gtdf = out@grna_target_data_frame)
}

cases <- list()
add <- function(label, gtdf, pairs, note) {
  r <- tryCatch(expand(gtdf, pairs), error = function(e) {
    cat(sprintf("  %-18s ERROR: %s\n", label, conditionMessage(e)))
    NULL
  })
  if (is.null(r)) return(invisible(NULL))
  cat(sprintf("  %-18s %d pairs in -> %d out\n", label, nrow(pairs), nrow(r$pairs)))
  cases[[length(cases) + 1]] <<- list(label = label, note = note,
                                      gtdf = gtdf, pairs_in = pairs, pairs_out = r$pairs,
                                      grna_group = r$gtdf$grna_group)
}

# A guide inside two overlapping elements, which is the shape real data has.
add("shared_guide",
    data.frame(grna_id = c("gA1", "gA2", "gS", "gS", "gB1", "ntc1", "ntc2"),
               grna_target = c("A", "A", "A", "B", "B", "non-targeting", "non-targeting"),
               stringsAsFactors = FALSE),
    data.frame(grna_target = c("A", "B", "A"), response_id = c("g1", "g1", "g2"),
               stringsAsFactors = FALSE),
    "gS belongs to A and B and must appear under each")

# A single-guide target, where the expansion is the identity.
add("one_guide_target",
    data.frame(grna_id = c("gC1", "ntc1"), grna_target = c("C", "non-targeting"),
               stringsAsFactors = FALSE),
    data.frame(grna_target = c("C"), response_id = c("g1"), stringsAsFactors = FALSE),
    "one row in, one row out")

# Non-targeting guides are in the table but no pair names them, so they must not appear.
add("ntcs_are_not_expanded",
    data.frame(grna_id = c("gA1", "ntc1", "ntc2", "ntc3"),
               grna_target = c("A", rep("non-targeting", 3)), stringsAsFactors = FALSE),
    data.frame(grna_target = "A", response_id = "g1", stringsAsFactors = FALSE),
    "grna_group stays 'non-targeting' for NTCs and they are not in the output")

# A pair naming a target with no guides in the table. Records whatever R does, which is the
# question this case exists to answer.
add("orphan_target",
    data.frame(grna_id = c("gA1", "ntc1"), grna_target = c("A", "non-targeting"),
               stringsAsFactors = FALSE),
    data.frame(grna_target = c("A", "MISSING"), response_id = c("g1", "g1"),
               stringsAsFactors = FALSE),
    "target absent from grna_target_data_frame")

json_str <- function(v) paste0('"', gsub('"', '\\\\"', as.character(v)), '"', collapse = ", ")
frame_json <- function(df) paste(vapply(names(df), function(n)
  sprintf('"%s": [%s]', n, json_str(df[[n]])), character(1)), collapse = ", ")

blocks <- vapply(cases, function(c) sprintf(
  '    {\n      "label": "%s",\n      "note": "%s",\n      "gtdf": {%s},\n      "pairs_in": {%s},\n      "pairs_out": {%s},\n      "grna_group": [%s]\n    }',
  c$label, c$note, frame_json(c$gtdf), frame_json(c$pairs_in), frame_json(c$pairs_out),
  json_str(c$grna_group)), character(1))

writeLines(c("{",
             sprintf('  "sceptre_version": "%s",', as.character(utils::packageVersion("sceptre"))),
             '  "cases": [', paste(blocks, collapse = ",\n"), "  ]", "}"), out_path)
cat(sprintf("wrote %s (%d cases, sceptre %s)\n", out_path, length(cases),
            as.character(utils::packageVersion("sceptre"))))
