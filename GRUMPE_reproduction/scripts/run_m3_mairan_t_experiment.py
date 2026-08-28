#!/usr/bin/env python3
"""Run the Mairan T three-image AMSA PHCL and constrained integration pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from grumpe_method.integration import integrate_gradients
from grumpe_method.photoclinometry import (
    centered_slopes_to_edges,
    estimate_phcl,
    render_hapke_stack,
)
from grumpe_method.reflectance import HapkeParameters


def write_tif(path: Path, data: np.ndarray, template: Path, nodata: float = -32768.0) -> None:
    with rasterio.open(template) as source:
        profile = source.profile.copy()
    profile.update(dtype="float32", count=1, nodata=nodata, compress="deflate", predictor=3)
    array = np.asarray(data, dtype=np.float32)
    with rasterio.open(path, "w", **profile) as target:
        target.write(np.where(np.isfinite(array), array, nodata), 1)


def image_metrics(observed: np.ndarray, modeled: np.ndarray, usable: np.ndarray) -> dict[str, float]:
    residual = modeled[usable] - observed[usable]
    centered_observed = observed[usable] - np.mean(observed[usable])
    centered_modeled = modeled[usable] - np.mean(modeled[usable])
    denominator = np.linalg.norm(centered_observed) * np.linalg.norm(centered_modeled)
    return {
        "pixels": int(residual.size),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "correlation": float(np.dot(centered_observed, centered_modeled) / denominator)
        if denominator > 0 else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-iterations", type=int, default=80)
    args = parser.parse_args()
    root = args.root.resolve()
    preprocessed = root / "02_preprocessed"
    output = root / "04_results" / "mairan_t_amsa"
    output.mkdir(parents=True, exist_ok=True)
    data = np.load(preprocessed / "mairan_t_m3_stack.npz")
    images = np.asarray(data["images"], dtype=np.float64)
    sun = np.asarray(data["sun_directions"], dtype=np.float64)
    view = np.asarray(data["view_directions"], dtype=np.float64)
    reference = np.asarray(data["initial_dem"], dtype=np.float64)
    common = np.asarray(data["common_mask"], dtype=bool)
    dx = float(data["pixel_size_x"])
    dy = float(data["pixel_size_y"])
    scene_ids = [str(value) for value in data["scene_ids"]]

    # Array rows grow southward, while the ENU view/sun y component points north.
    q_row, p0 = np.gradient(reference, dy, dx)
    q0 = -q_row
    parameters = HapkeParameters(
        single_scattering_albedo=0.4,
        opposition_amplitude=1.0,
        opposition_width=0.06,
        phase_function="dhg",
        dhg_b=0.30,
        dhg_c=0.70,
        roughness_degrees=11.0,
        h_function="level2",
        legendre_order=15,
    )
    initial_albedo = np.full(reference.shape, 0.4)
    initial_model = render_hapke_stack(
        p0, q0, initial_albedo, sun, view, parameters, "amsa"
    )
    usable = common[None, ...] & np.isfinite(images) & np.isfinite(initial_model)
    # Exclude only the darkest one percent per scene as likely unresolved cast shadow.
    observed = images.copy()
    shadow_thresholds: dict[str, float] = {}
    for index, scene in enumerate(scene_ids):
        threshold = float(np.percentile(images[index][usable[index]], 1.0))
        shadow_thresholds[scene] = threshold
        observed[index][~(usable[index] & (images[index] > threshold))] = np.nan

    print("Running AMSA PHCL with paper Mairan T parameters", flush=True)
    result = estimate_phcl(
        observed,
        sun,
        view,
        reference,
        pixel_size_x=dx,
        pixel_size_y=dy,
        parameters=parameters,
        reflectance_model="amsa",
        reference_p=p0,
        reference_q=q0,
        dem_weight=1.0e-4,
        dem_sigma_px=15.0,
        albedo_sigma_px=21.0,
        initial_albedo=0.4,
        max_iterations=args.max_iterations,
        relative_tolerance=1.0e-6,
        max_rejections=50,
    )

    # Convert north-positive centered q to south-positive row-edge q.  The paper
    # reports s=100 on a pixel-coordinate grid, so use height change per pixel
    # and unit pixel spacing here rather than applying 100 to metre derivatives.
    p_edge_physical, q_north_edge = centered_slopes_to_edges(result.p, result.q)
    p_edge_pixel = p_edge_physical * dx
    q_row_edge_pixel = -q_north_edge * dy
    print("Integrating gradients: Grumpe s=100, r_DEM=15 px", flush=True)
    grumpe = integrate_gradients(
        p_edge_pixel,
        q_row_edge_pixel,
        reference,
        pixel_size_x=1.0,
        pixel_size_y=1.0,
        depth_weight=100.0,
        lowpass_sigma_px=15.0,
        rtol=1.0e-9,
        max_iterations=5000,
    )
    print("Integrating gradients: Horn/Poisson control", flush=True)
    horn = integrate_gradients(
        p_edge_pixel,
        q_row_edge_pixel,
        reference,
        pixel_size_x=1.0,
        pixel_size_y=1.0,
        depth_weight=0.0,
        lowpass_sigma_px=0.0,
        rtol=1.0e-9,
        max_iterations=5000,
    )
    # Remove the arbitrary Poisson datum using the reference mean.
    horn_dem = horn.dem + np.mean(reference - horn.dem)

    final_modeled = result.modeled_images
    initial_metrics, final_metrics = {}, {}
    for index, scene in enumerate(scene_ids):
        mask = np.isfinite(observed[index]) & np.isfinite(final_modeled[index])
        initial_metrics[scene] = image_metrics(observed[index], initial_model[index], mask)
        final_metrics[scene] = image_metrics(observed[index], final_modeled[index], mask)

    template = preprocessed / "GLD100_MairanT_300ppd.tif"
    write_tif(output / "MairanT_GRUMPE_AMSA_DEM.tif", grumpe.dem, template)
    write_tif(output / "MairanT_Horn_Poisson_DEM.tif", horn_dem, template)
    write_tif(output / "MairanT_GRUMPE_minus_GLD100.tif", grumpe.dem - reference, template)
    write_tif(output / "MairanT_Horn_minus_GLD100.tif", horn_dem - reference, template)
    write_tif(output / "MairanT_single_scattering_albedo.tif", result.single_scattering_albedo, template)
    write_tif(output / "MairanT_p_east.tif", result.p, template)
    write_tif(output / "MairanT_q_north.tif", result.q, template)
    for index, scene in enumerate(scene_ids):
        write_tif(output / f"{scene}_AMSA_modeled_reflectance.tif", final_modeled[index], template)
        write_tif(output / f"{scene}_AMSA_residual.tif", final_modeled[index] - observed[index], template)

    np.savez_compressed(
        output / "mairan_t_amsa_result.npz",
        p=result.p,
        q_north=result.q,
        albedo=result.single_scattering_albedo,
        modeled_images=final_modeled,
        valid_mask=result.valid_mask,
        grumpe_dem=grumpe.dem,
        horn_dem=horn_dem,
        initial_dem=reference,
    )
    (output / "phcl_history.json").write_text(
        json.dumps(list(result.history), indent=2), encoding="utf-8"
    )

    delta = grumpe.dem - reference
    horn_delta = horn_dem - reference
    accepted = int(sum(row["accepted"] != 0.0 for row in result.history))
    metrics = {
        "status": "completed",
        "experiment_scope": "Mairan T PHCL plus constrained gradient integration; extended SfS and light-independent registration are not yet implemented",
        "scene_ids": scene_ids,
        "paper_parameters": {
            "reflectance_model": "AMSA",
            "phcl_dem_weight_d": 1.0e-4,
            "phcl_dem_gaussian_width_px": 15.0,
            "albedo_gaussian_width_px": 21.0,
            "integration_weight_s": 100.0,
            "integration_dem_gaussian_width_px": 15.0,
            "roughness_degrees": 11.0,
        },
        "implementation_notes": {
            "integration": "same quadratic objective solved by matrix-free conjugate gradients",
            "integration_coordinates": "unit pixel coordinates; PHCL physical slopes converted to metres per pixel",
            "fixed_hapke_parameters": "current project Warell-like parameter set; exact Warell (2004) solution table remains an external-source verification item",
            "shadow_handling": "lowest 1 percent of reflectance per scene excluded",
        },
        "shadow_thresholds": shadow_thresholds,
        "phcl": {
            "iterations": len(result.history),
            "accepted_steps": accepted,
            "converged": bool(result.converged),
            "valid_pixels": int(result.valid_mask.sum()),
            "initial_total_error": float(result.history[0]["total_error"]),
            "final_total_error": float(result.history[-1]["total_error"]),
        },
        "image_fit_initial": initial_metrics,
        "image_fit_final": final_metrics,
        "grumpe_integration": {
            "converged": bool(grumpe.converged),
            "iterations": int(grumpe.iterations),
            "relative_residual": grumpe.relative_residual,
            "delta_min_m": float(np.min(delta)),
            "delta_max_m": float(np.max(delta)),
            "delta_mean_m": float(np.mean(delta)),
            "delta_std_m": float(np.std(delta)),
            "delta_rmse_m": float(np.sqrt(np.mean(delta**2))),
        },
        "horn_poisson": {
            "converged": bool(horn.converged),
            "iterations": int(horn.iterations),
            "relative_residual": horn.relative_residual,
            "delta_min_m": float(np.min(horn_delta)),
            "delta_max_m": float(np.max(horn_delta)),
            "delta_std_m": float(np.std(horn_delta)),
        },
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    panels = (
        (reference, "GLD100 initial DEM", "terrain"),
        (grumpe.dem, "Grumpe constrained DEM", "terrain"),
        (horn_dem, "Horn/Poisson DEM", "terrain"),
        (delta, "Grumpe - GLD100 (m)", "RdBu_r"),
        (horn_delta, "Horn - GLD100 (m)", "RdBu_r"),
        (result.single_scattering_albedo, "Single-scattering albedo", "viridis"),
    )
    for axis, (array, title, cmap) in zip(axes.flat, panels):
        if "minus" in title.lower() or " - " in title:
            limit = np.percentile(np.abs(array), 99)
            image = axis.imshow(array, cmap=cmap, vmin=-limit, vmax=limit)
        else:
            image = axis.imshow(array, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
        fig.colorbar(image, ax=axis, shrink=0.75)
    fig.savefig(output / "dem_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
    for index, scene in enumerate(scene_ids):
        finite = observed[index][np.isfinite(observed[index])]
        vmin, vmax = np.percentile(finite, [1, 99])
        axes[index, 0].imshow(observed[index], cmap="gray", vmin=vmin, vmax=vmax)
        axes[index, 1].imshow(final_modeled[index], cmap="gray", vmin=vmin, vmax=vmax)
        residual = final_modeled[index] - observed[index]
        limit = np.nanpercentile(np.abs(residual), 99)
        axes[index, 2].imshow(residual, cmap="RdBu_r", vmin=-limit, vmax=limit)
        axes[index, 0].set_ylabel(scene)
    for axis, title in zip(axes[0], ("Observed L/E", "AMSA modeled", "Residual")):
        axis.set_title(title)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.savefig(output / "reflectance_fit.png", dpi=180)
    plt.close(fig)
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
