#!/usr/bin/env python3
"""Compare Grumpe's Hapke IMSA and AMSA choices under controlled mismatch."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from grumpe_method.photoclinometry import estimate_phcl
from grumpe_method.reflectance import HapkeParameters
from grumpe_method.relaxation import solve_paper_relaxation
from grumpe_method.synthetic import make_synthetic_case


def _rmse(estimate: np.ndarray, truth: np.ndarray, mask: np.ndarray | None = None) -> float:
    active = np.isfinite(estimate) & np.isfinite(truth)
    if mask is not None:
        active &= mask
    return float(np.sqrt(np.mean((estimate[active] - truth[active]) ** 2)))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parameters(config: dict[str, object], roughness: float) -> HapkeParameters:
    return HapkeParameters(
        opposition_amplitude=float(config["opposition_amplitude"]),
        opposition_width=float(config["opposition_width"]),
        phase_function=str(config["phase_function"]),
        dhg_b=float(config["dhg_b"]),
        dhg_c=float(config["dhg_c"]),
        roughness_degrees=float(roughness),
        h_function=str(config["h_function"]),
        legendre_order=int(config["legendre_order"]),
    )


def _render_figures(output: Path, rows: list[dict[str, object]], arrays: dict[str, np.ndarray]) -> None:
    labels = [str(row["id"]) for row in rows]
    positions = np.arange(len(rows))

    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    axes[0].bar(positions, [float(row["dem_rmse_m"]) for row in rows], color="#3b82f6")
    axes[0].axhline(float(rows[0]["initial_dem_rmse_m"]), color="black", ls="--", label="Initial DEM")
    axes[0].set_ylabel("DEM RMSE (m)")
    axes[0].legend()
    axes[1].bar(positions, [float(row["slope_vector_rmse"]) for row in rows], color="#10b981")
    axes[1].set_ylabel("Slope-vector RMSE")
    axes[2].bar(positions, [float(row["albedo_rmse"]) for row in rows], color="#f59e0b")
    axes[2].set_ylabel("Single-scattering albedo RMSE")
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Controlled Hapke model comparison (all non-model inputs fixed)")
    figure.savefig(output / "model_metric_comparison.png", dpi=180)
    plt.close(figure)

    truth = arrays["truth_dem"]
    initial = arrays["initial_dem"]
    error_limit = max(
        float(np.nanpercentile(np.abs(arrays[f"{label}_dem"] - truth), 99))
        for label in labels
    )
    panels: list[tuple[str, np.ndarray, str, float | None, float | None]] = [
        ("Truth DEM", truth, "terrain", None, None),
        ("Initial DEM error", initial - truth, "coolwarm", -error_limit, error_limit),
    ]
    panels.extend(
        (label, arrays[f"{label}_dem"] - truth, "coolwarm", -error_limit, error_limit)
        for label in labels
    )
    columns = 3
    rows_count = int(np.ceil(len(panels) / columns))
    figure, axes = plt.subplots(
        rows_count, columns, figsize=(15, 4.5 * rows_count), constrained_layout=True
    )
    flat_axes = np.asarray(axes).flat
    for axis, (title, values, cmap, vmin, vmax) in zip(flat_axes, panels):
        image = axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_axis_off()
        figure.colorbar(image, ax=axis, shrink=0.75)
    for axis in list(flat_axes):
        axis.set_axis_off()
    figure.suptitle("DEM truth and reconstruction error maps (m)")
    figure.savefig(output / "dem_error_maps.png", dpi=180)
    plt.close(figure)

    model_delta = arrays["amsa11_images"] - arrays["imsa11_images"]
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    limit = float(np.nanpercentile(np.abs(model_delta), 99))
    for index, axis in enumerate(axes.flat):
        image = axis.imshow(model_delta[index], cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.set_title(f"Observation {index + 1}: AMSA - IMSA")
        axis.set_axis_off()
        figure.colorbar(image, ax=axis, shrink=0.75)
    figure.suptitle("Reflectance-model difference at identical terrain/albedo/roughness")
    figure.savefig(output / "amsa_minus_imsa_reflectance.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/reflectance_model_comparison.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reflectance_model_comparison"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed experiment rows and NPY arrays in the output directory",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)

    case_config = config["case"]
    hapke_config = config["hapke"]
    phcl_config = config["phcl"]
    integration_config = config["integration"]
    experiments = config["experiments"]
    metrics_json = args.output / "comparison_metrics.json"
    rows: list[dict[str, object]] = (
        json.loads(metrics_json.read_text(encoding="utf-8"))
        if args.resume and metrics_json.exists()
        else []
    )
    arrays: dict[str, np.ndarray] = {}
    if args.resume:
        for path in args.output.glob("*.npy"):
            arrays[path.stem] = np.load(path)
    completed_ids = {str(row["id"]) for row in rows}

    for index, experiment in enumerate(experiments, start=1):
        experiment_id = str(experiment["id"])
        if experiment_id in completed_ids:
            print(f"[{index}/{len(experiments)}] {experiment_id}: reused", flush=True)
            continue
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
            seed=int(case_config["seed"]),
            parameters=truth_parameters,
            reflectance_model=str(experiment["truth_model"]),
        )
        print(f"[{index}/{len(experiments)}] {experiment_id}: PHCL", flush=True)
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
        print(f"[{index}/{len(experiments)}] {experiment_id}: paper integration", flush=True)
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
        active_images = np.isfinite(case.images) & np.isfinite(phcl.modeled_images)
        p_error = phcl.p - case.truth_p
        q_error = phcl.q - case.truth_q
        row: dict[str, object] = {
            "id": experiment_id,
            "category": str(experiment["category"]),
            "truth_model": str(experiment["truth_model"]),
            "inversion_model": str(experiment["inversion_model"]),
            "truth_roughness_degrees": float(experiment["truth_roughness_degrees"]),
            "inversion_roughness_degrees": float(experiment["inversion_roughness_degrees"]),
            "phcl_converged": bool(phcl.converged),
            "phcl_iterations": len(phcl.history),
            "valid_pixel_fraction": float(np.mean(phcl.valid_mask)),
            "reflectance_rmse": _rmse(phcl.modeled_images, case.images, active_images),
            "p_rmse": _rmse(phcl.p, case.truth_p, phcl.valid_mask),
            "q_rmse": _rmse(phcl.q, case.truth_q, phcl.valid_mask),
            "slope_vector_rmse": float(
                np.sqrt(np.mean((p_error[phcl.valid_mask] ** 2 + q_error[phcl.valid_mask] ** 2)))
            ),
            "albedo_rmse": _rmse(
                phcl.single_scattering_albedo, case.truth_w, phcl.valid_mask
            ),
            "initial_dem_rmse_m": _rmse(case.initial_dem, case.truth_dem),
            "dem_rmse_m": _rmse(integrated.dem, case.truth_dem),
            "dem_mean_bias_m": float(np.mean(integrated.dem - case.truth_dem)),
            "integration_best_energy": float(integrated.best_error),
            "integration_successful_iterations": int(integrated.successful_iterations),
            "integration_total_updates": int(integrated.total_updates),
            "integration_stop_reason": integrated.stop_reason,
        }
        rows.append(row)
        arrays[f"{experiment_id}_dem"] = integrated.dem
        arrays[f"{experiment_id}_p"] = phcl.p
        arrays[f"{experiment_id}_q"] = phcl.q
        arrays[f"{experiment_id}_w"] = phcl.single_scattering_albedo
        arrays["truth_dem"] = case.truth_dem
        arrays["initial_dem"] = case.initial_dem
        if experiment_id == "imsa11_to_imsa11":
            arrays["imsa11_images"] = case.images
        if experiment_id == "amsa11_to_amsa11":
            arrays["amsa11_images"] = case.images
        print(
            f"[{index}/{len(experiments)}] {experiment_id}: "
            f"DEM RMSE={row['dem_rmse_m']:.6f} m, "
            f"slope RMSE={row['slope_vector_rmse']:.6f}",
            flush=True,
        )

    configured_order = {str(item["id"]): index for index, item in enumerate(experiments)}
    rows.sort(key=lambda row: configured_order[str(row["id"])])
    _write_csv(args.output / "comparison_metrics.csv", rows)
    (args.output / "comparison_metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name, values in arrays.items():
        np.save(args.output / f"{name}.npy", values)
    _render_figures(args.output, rows, arrays)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
