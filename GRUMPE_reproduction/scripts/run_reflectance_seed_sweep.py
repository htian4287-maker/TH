#!/usr/bin/env python3
"""Repeat the IMSA/AMSA matched and mismatched comparison across noise seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from grumpe_method.photoclinometry import estimate_phcl
from grumpe_method.relaxation import solve_paper_relaxation
from grumpe_method.synthetic import make_synthetic_case
from run_reflectance_model_comparison import _parameters, _rmse, _write_csv


METRICS = ("dem_rmse_m", "slope_vector_rmse", "albedo_rmse", "reflectance_rmse")


def _summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for experiment_id in dict.fromkeys(str(row["id"]) for row in rows):
        selected = [row for row in rows if row["id"] == experiment_id]
        summary: dict[str, object] = {
            "id": experiment_id,
            "truth_model": selected[0]["truth_model"],
            "inversion_model": selected[0]["inversion_model"],
            "runs": len(selected),
        }
        for metric in METRICS:
            values = np.array([float(row[metric]) for row in selected])
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values, ddof=1))
        summaries.append(summary)
    return summaries


def _paired_differences(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    comparisons = (
        ("amsa_truth", "amsa11_to_amsa11", "amsa11_to_imsa11"),
        ("imsa_truth", "imsa11_to_imsa11", "imsa11_to_amsa11"),
    )
    results: list[dict[str, object]] = []
    for label, matched_id, mismatch_id in comparisons:
        seeds = sorted({int(row["seed"]) for row in rows if row["id"] == matched_id})
        for metric in METRICS:
            deltas = []
            for seed in seeds:
                matched = next(
                    float(row[metric])
                    for row in rows
                    if row["id"] == matched_id and int(row["seed"]) == seed
                )
                mismatched = next(
                    float(row[metric])
                    for row in rows
                    if row["id"] == mismatch_id and int(row["seed"]) == seed
                )
                deltas.append(mismatched - matched)
            results.append(
                {
                    "truth_case": label,
                    "metric": metric,
                    "mismatch_minus_matched_mean": float(np.mean(deltas)),
                    "mismatch_minus_matched_std": float(np.std(deltas, ddof=1)),
                    "mismatch_worse_count": int(np.sum(np.asarray(deltas) > 0.0)),
                    "runs": len(deltas),
                }
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("config/reflectance_model_comparison.yaml")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/reflectance_model_comparison")
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    case_config = config["case"]
    hapke_config = config["hapke"]
    phcl_config = config["phcl"]
    integration_config = config["integration"]
    experiments = config["experiments"][:4]
    seeds = [int(value) for value in case_config["seed_sweep"]]
    rows: list[dict[str, object]] = []
    total = len(seeds) * len(experiments)
    counter = 0

    for seed in seeds:
        for experiment in experiments:
            counter += 1
            experiment_id = str(experiment["id"])
            truth_parameters = _parameters(
                hapke_config, float(experiment["truth_roughness_degrees"])
            )
            inversion_parameters = _parameters(
                hapke_config, float(experiment["inversion_roughness_degrees"])
            )
            case = make_synthetic_case(
                size=int(case_config["size"]),
                pixel_size_m=float(case_config["pixel_size_m"]),
                prior_sigma_px=float(case_config["prior_sigma_px"]),
                noise_std=float(case_config["noise_std"]),
                seed=seed,
                parameters=truth_parameters,
                reflectance_model=str(experiment["truth_model"]),
            )
            phcl = estimate_phcl(
                case.images,
                case.sun_directions,
                case.view_directions,
                case.initial_dem,
                pixel_size_x=case.pixel_size_m,
                pixel_size_y=case.pixel_size_m,
                parameters=inversion_parameters,
                reflectance_model=str(experiment["inversion_model"]),
                dem_weight=float(phcl_config["dem_weight"]),
                dem_sigma_px=float(phcl_config["dem_sigma_px"]),
                initial_albedo=float(phcl_config["initial_albedo"]),
                max_iterations=int(phcl_config["max_iterations"]),
                relative_tolerance=float(phcl_config["relative_tolerance"]),
                max_rejections=int(phcl_config["max_rejections"]),
            )
            integrated = solve_paper_relaxation(
                phcl.p,
                phcl.q,
                case.initial_dem,
                pixel_size_x=case.pixel_size_m,
                pixel_size_y=case.pixel_size_m,
                depth_weight=float(integration_config["depth_weight"]),
                lowpass_sigma_px=float(integration_config["lowpass_sigma_px"]),
                max_successful_iterations=int(integration_config["max_successful_iterations"]),
                max_steps_without_improvement=int(integration_config["max_steps_without_improvement"]),
                max_total_updates=int(integration_config["max_total_updates"]),
                relative_tolerance=float(integration_config["relative_tolerance"]),
            )
            active = np.isfinite(case.images) & np.isfinite(phcl.modeled_images)
            p_error = phcl.p - case.truth_p
            q_error = phcl.q - case.truth_q
            row: dict[str, object] = {
                "seed": seed,
                "id": experiment_id,
                "truth_model": str(experiment["truth_model"]),
                "inversion_model": str(experiment["inversion_model"]),
                "phcl_converged": bool(phcl.converged),
                "dem_rmse_m": _rmse(integrated.dem, case.truth_dem),
                "slope_vector_rmse": float(
                    np.sqrt(np.mean(p_error[phcl.valid_mask] ** 2 + q_error[phcl.valid_mask] ** 2))
                ),
                "albedo_rmse": _rmse(
                    phcl.single_scattering_albedo, case.truth_w, phcl.valid_mask
                ),
                "reflectance_rmse": _rmse(phcl.modeled_images, case.images, active),
            }
            rows.append(row)
            print(
                f"[{counter}/{total}] seed={seed} {experiment_id}: "
                f"DEM={row['dem_rmse_m']:.6f}, albedo={row['albedo_rmse']:.6f}",
                flush=True,
            )

    summaries = _summarize(rows)
    paired = _paired_differences(rows)
    _write_csv(args.output / "seed_sweep_runs.csv", rows)
    _write_csv(args.output / "seed_sweep_summary.csv", summaries)
    _write_csv(args.output / "seed_sweep_paired_differences.csv", paired)
    (args.output / "seed_sweep.json").write_text(
        json.dumps({"runs": rows, "summary": summaries, "paired": paired}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    labels = [str(item["id"]) for item in summaries]
    figure, axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
    for axis, metric in zip(axes, METRICS, strict=True):
        values = [
            [float(row[metric]) for row in rows if row["id"] == experiment_id]
            for experiment_id in labels
        ]
        axis.boxplot(values, tick_labels=labels, showmeans=True)
        axis.set_title(metric)
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Five-seed robustness check: matched vs mismatched multiple scattering")
    figure.savefig(args.output / "seed_sweep_boxplots.png", dpi=180)
    plt.close(figure)
    print(f"Seed sweep written to {args.output}")


if __name__ == "__main__":
    main()
