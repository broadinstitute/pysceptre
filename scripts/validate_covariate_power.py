"""Validate analytical power against simulated pysceptre discovery rejections.

Run from an installed checkout; output contains synthetic results only.
Example: python scripts/validate_covariate_power.py --blocks 20 --replicates-per-block 100 --jobs 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm, t
from threadpoolctl import threadpool_limits

from pysceptre import compute_power, run_discovery_analysis

SCENARIOS = {
    "randomized_batches": (0.5, 0.5),
    "moderate_confounding": (0.25, 0.75),
    "strong_confounding": (0.1, 0.9),
    "continuous_confounding": None,
}


def _design(scenario, n_cells, batch_baseline_means):
    if scenario == "continuous_confounding":
        z = np.linspace(-2, 2, n_cells)
        return np.column_stack([np.ones(n_cells), z]), np.array([np.log(2), 0.6])
    batch = np.repeat([0.0, 1.0], n_cells // 2)
    return np.column_stack([np.ones(n_cells), batch]), np.array(
        [np.log(batch_baseline_means[0]), np.log(batch_baseline_means[1] / batch_baseline_means[0])]
    )


def _assignment(scenario, Z, rng):
    n = len(Z)
    if scenario == "continuous_confounding":
        return np.flatnonzero(rng.random(n) < expit(1.5 * Z[:, 1]))
    probabilities = SCENARIOS[scenario]
    return np.r_[
        rng.choice(n // 2, round(probabilities[0] * n / 2), replace=False),
        n // 2 + rng.choice(n // 2, round(probabilities[1] * n / 2), replace=False),
    ]


def _case(args, scenario, fold_change, case_id):
    start = time.perf_counter()
    Z, coefs = _design(scenario, args.n_cells, args.batch_baseline_means)
    baseline = np.exp(Z @ coefs)
    theta = 2.0
    prediction_pair = pd.DataFrame({"grna_target": ["target"], "response_id": ["g"]})
    records, blocks = [], []
    with threadpool_limits(limits=1):
        for block in range(args.blocks):
            children = np.random.SeedSequence([args.seed, case_id, block]).spawn(3)
            assignment_rng, count_rng = [np.random.default_rng(s) for s in children[:2]]
            crt_seed = int(children[2].generate_state(1)[0])
            idx = _assignment(scenario, Z, assignment_rng)
            mu1 = baseline.copy()
            mu1[idx] *= fold_change
            Y = count_rng.negative_binomial(
                theta, theta / (theta + mu1), size=(args.replicates_per_block, args.n_cells)
            ).astype(float)
            genes = [f"rep_{j}" for j in range(args.replicates_per_block)]
            pairs = pd.DataFrame({"response_id": genes, "grna_target": "target"})
            observed = run_discovery_analysis(
                response_matrix=Y,
                gene_ids=genes,
                covariate_matrix=Z,
                grna_target_cells={"target": idx},
                pairs=pairs,
                side="left",
                moi="high",
                control_group="complement",
                resampling_mechanism="crt",
                resampling_approximation="skew_normal",
                seed=crt_seed,
                n_jobs=1,
            )
            if (
                len(observed) != len(genes)
                or not np.isfinite(observed[["p_value", "z_orig"]]).all().all()
            ):
                raise RuntimeError("discovery returned a missing or non-finite test result")
            for row in observed.itertuples(index=False):
                records.append(
                    {
                        "scenario": scenario,
                        "fold_change": fold_change,
                        "block": block,
                        "replicate": row.response_id,
                        "p_value": row.p_value,
                        "z_orig": row.z_orig,
                        "stage": int(row.stage),
                        "crt_seed": crt_seed,
                    }
                )
            for cutoff in args.cutoffs:
                common = dict(
                    fold_change_mean=fold_change,
                    fold_change_sd=0.0,
                    cutoff=cutoff,
                    num_total_cells=args.n_cells,
                    side="left",
                )
                adjusted = compute_power(
                    prediction_pair,
                    method="covariate",
                    covariate_matrix=Z,
                    gene_fits={"g": SimpleNamespace(fitted_coefs=coefs, theta=theta)},
                    grna_target_cells={"target": idx},
                    **common,
                ).iloc[0]
                homogeneous = compute_power(
                    prediction_pair,
                    pd.DataFrame({"grna_target": ["target"], "num_cells": [len(idx)]}),
                    pd.DataFrame(
                        {
                            "response_id": ["g"],
                            "expression_mean": [baseline.mean()],
                            "expression_size": [theta],
                        }
                    ),
                    **common,
                ).iloc[0]
                blocks.append(
                    {
                        "scenario": scenario,
                        "fold_change": fold_change,
                        "block": block,
                        "cutoff": cutoff,
                        "n_replicates": len(genes),
                        "n_treated": len(idx),
                        "empirical_power": float(observed.p_value.le(cutoff).mean()),
                        "normal_cutoff_rate": float(observed.z_orig.le(norm.ppf(cutoff)).mean()),
                        "covariate_power": float(adjusted.power),
                        "homogeneous_power": float(homogeneous.power),
                        "predicted_mean": float(adjusted.mean_test_stat),
                        "predicted_sd": float(adjusted.sd_test_stat),
                        "observed_mean": float(observed.z_orig.mean()),
                        "observed_second_moment": float(np.mean(observed.z_orig**2)),
                    }
                )
    return records, blocks, time.perf_counter() - start


def _interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - half, mean + half


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/covariate_power_validation"))
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--replicates-per-block", type=int, default=100)
    parser.add_argument("--n-cells", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--cutoffs", type=float, nargs="+", default=[0.05, 0.005])
    parser.add_argument("--fold-changes", type=float, nargs="+", default=[1.0, 0.85])
    parser.add_argument("--batch-baseline-means", type=float, nargs=2, default=[1.0, 4.0])
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
    args = parser.parse_args()
    if args.blocks < 2 or args.replicates_per_block < 2 or args.n_cells < 20 or args.n_cells % 2:
        parser.error("need >=2 blocks, >=2 replicates per block and an even n_cells>=20")
    if (
        args.jobs < 1
        or any(not 0 < c < 1 for c in args.cutoffs)
        or any(f <= 0 for f in args.fold_changes)
    ):
        parser.error("jobs must be positive; cutoffs in (0,1); fold changes positive")
    if any(not np.isfinite(m) or m <= 0 for m in args.batch_baseline_means):
        parser.error("batch baseline means must be finite and positive")
    args.output.mkdir(parents=True, exist_ok=True)
    rows, block_rows = [], []
    start = time.perf_counter()
    cases = [(s, f) for s in args.scenarios for f in args.fold_changes]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(_case, args, s, f, i): (s, f) for i, (s, f) in enumerate(cases)}
        for future in as_completed(futures):
            records, blocks, wall = future.result()
            rows.extend(records)
            block_rows.extend(blocks)
            scenario, fold_change = futures[future]
            print(
                f"completed {scenario}, fold_change={fold_change}: {len(records)} tests in {wall:.1f}s",
                flush=True,
            )
    replicates = pd.DataFrame(rows).sort_values(["scenario", "fold_change", "block", "replicate"])
    block_results = pd.DataFrame(block_rows).sort_values(
        ["scenario", "fold_change", "cutoff", "block"]
    )
    summaries = []
    for keys, group in block_results.groupby(["scenario", "fold_change", "cutoff"], sort=True):
        empirical, lo, hi = _interval(group.empirical_power)
        error, error_lo, error_hi = _interval(group.empirical_power - group.covariate_power)
        observed_mean = float(group.observed_mean.mean())
        sd = np.sqrt(group.observed_second_moment.mean() - observed_mean**2)
        predicted_mean = float(group.predicted_mean.mean())
        predicted_sd = np.sqrt(
            np.mean(group.predicted_sd**2 + group.predicted_mean**2) - predicted_mean**2
        )
        summaries.append(
            dict(
                scenario=keys[0],
                fold_change=keys[1],
                cutoff=keys[2],
                n_replicates=int(group.n_replicates.sum()),
                empirical_power=empirical,
                empirical_ci_low=max(0, lo),
                empirical_ci_high=min(1, hi),
                covariate_power=float(group.covariate_power.mean()),
                homogeneous_power=float(group.homogeneous_power.mean()),
                empirical_minus_covariate=error,
                difference_ci_low=error_lo,
                difference_ci_high=error_hi,
                normal_cutoff_rate=float(group.normal_cutoff_rate.mean()),
                observed_stat_mean=observed_mean,
                observed_stat_sd=float(sd),
                predicted_stat_mean=predicted_mean,
                predicted_stat_sd=float(predicted_sd),
            )
        )
    summary = pd.DataFrame(summaries)
    replicates.to_csv(args.output / "replicates.tsv", sep="\t", index=False)
    block_results.to_csv(args.output / "blocks.tsv", sep="\t", index=False)
    summary.to_csv(args.output / "summary.tsv", sep="\t", index=False)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "--", "src/pysceptre/analytical_power"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    (args.output / "tracked_source.diff").write_text(diff)
    metadata = dict(
        seed=args.seed,
        blocks=args.blocks,
        replicates_per_block=args.replicates_per_block,
        n_cells=args.n_cells,
        cutoffs=args.cutoffs,
        fold_changes=args.fold_changes,
        baseline_batch_means=args.batch_baseline_means,
        scenarios=args.scenarios,
        theta=2.0,
        continuous_baseline_coefs=[float(np.log(2)), 0.6],
        continuous_assignment="expit(1.5*z), z=linspace(-2,2,n_cells)",
        side="left",
        resampling_mechanism="crt",
        resampling_approximation="skew_normal",
        baseline_inputs="oracle unperturbed coefficients and NB size; not estimated from effect-bearing data",
        uncertainty="95% Student-t intervals across independent blocks; each block shares CRT draws across its count replicates",
        conditional_design="batch counts fixed; cell indices redrawn independently by block; continuous assignments are Bernoulli",
        wall_seconds=time.perf_counter() - start,
        head=head,
        python=platform.python_version(),
        packages={p: version(p) for p in ["pysceptre", "numpy", "scipy", "pandas"]},
        stage_counts={str(k): int(v) for k, v in replicates.stage.value_counts().items()},
    )
    source_dir = Path(__file__).resolve().parents[1] / "src" / "pysceptre" / "analytical_power"
    snapshot_dir = args.output / "source_snapshot"
    snapshot_dir.mkdir(exist_ok=True)
    metadata["source_sha256"] = {}
    for source in [Path(__file__).resolve(), *sorted(source_dir.glob("*.py"))]:
        payload = source.read_bytes()
        metadata["source_sha256"][source.name] = hashlib.sha256(payload).hexdigest()
        (snapshot_dir / source.name).write_bytes(payload)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        summary[
            [
                "scenario",
                "fold_change",
                "cutoff",
                "empirical_power",
                "empirical_ci_low",
                "empirical_ci_high",
                "covariate_power",
                "homogeneous_power",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(f"total wall time {metadata['wall_seconds']:.1f}s; output {args.output}", flush=True)


if __name__ == "__main__":
    main()
