#!/usr/bin/env python3
"""Run the seven missing parts of the Mairan T Grumpe reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter

from grumpe_method.extended_sfs import estimate_extended_sfs
from grumpe_method.integration import integrate_gradients
from grumpe_method.photoclinometry import centered_slopes_to_edges, estimate_phcl
from grumpe_method.reflectance import warell_moon_2004
from grumpe_method.registration import register_by_single_image_phcl
from grumpe_method.relaxation import solve_paper_relaxation


def write_tif(path: Path, values: np.ndarray, template: Path, nodata: float = -32768.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(template) as source:
        profile = source.profile.copy()
    profile.update(dtype="float32", count=1, nodata=nodata, compress="deflate", predictor=3)
    array = np.asarray(values, dtype=np.float32)
    with rasterio.open(path, "w", **profile) as target:
        target.write(np.where(np.isfinite(array), array, nodata), 1)


def image_fit(observed: np.ndarray, modeled: np.ndarray) -> dict[str, float]:
    usable = np.isfinite(observed) & np.isfinite(modeled)
    residual = modeled[usable] - observed[usable]
    x = observed[usable] - np.mean(observed[usable])
    y = modeled[usable] - np.mean(modeled[usable])
    return {
        "pixels": int(residual.size),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "correlation": float(np.dot(x, y) / max(np.linalg.norm(x) * np.linalg.norm(y), 1.0e-15)),
    }


def prepare_observed(images: np.ndarray, common: np.ndarray) -> tuple[np.ndarray, list[float]]:
    observed = np.asarray(images, dtype=np.float64).copy()
    thresholds = []
    for index in range(observed.shape[0]):
        valid = common & np.isfinite(observed[index])
        threshold = float(np.percentile(observed[index][valid], 1.0))
        thresholds.append(threshold)
        observed[index][~(valid & (observed[index] > threshold))] = np.nan
    return observed, thresholds


def integrate_cg(p: np.ndarray, q_north: np.ndarray, reference: np.ndarray, dx: float, dy: float):
    p_edge, q_north_edge = centered_slopes_to_edges(p, q_north)
    return integrate_gradients(
        p_edge * dx,
        -q_north_edge * dy,
        reference,
        pixel_size_x=1.0,
        pixel_size_y=1.0,
        depth_weight=100.0,
        lowpass_sigma_px=15.0,
        rtol=1.0e-9,
        max_iterations=5000,
    )


def albedo_topography_leakage(albedo: np.ndarray, dem: np.ndarray, mask: np.ndarray) -> float:
    highpass = dem - gaussian_filter(dem, 7.0, mode="reflect")
    valid = mask & np.isfinite(albedo) & np.isfinite(highpass)
    a = albedo[valid] - np.mean(albedo[valid])
    h = highpass[valid] - np.mean(highpass[valid])
    return float(abs(np.dot(a, h) / max(np.linalg.norm(a) * np.linalg.norm(h), 1.0e-15)))


def run_phcl(
    observed: np.ndarray,
    sun: np.ndarray,
    view: np.ndarray,
    reference: np.ndarray,
    dx: float,
    dy: float,
    model: str,
):
    return estimate_phcl(
        observed, sun, view, reference,
        pixel_size_x=dx, pixel_size_y=dy,
        parameters=warell_moon_2004(), reflectance_model=model,
        dem_weight=1.0e-4, dem_sigma_px=15.0, albedo_sigma_px=21.0,
        initial_albedo=0.4, max_iterations=160,
        relative_tolerance=1.0e-6, max_rejections=50,
    )


def run_sfs_cycles(
    observed: np.ndarray,
    sun: np.ndarray,
    view: np.ndarray,
    reference: np.ndarray,
    dx: float,
    dy: float,
    phcl,
    model: str,
    sigmas: list[float],
    iterations_per_cycle: int,
) -> tuple[object, list[dict[str, float]]]:
    cg = integrate_cg(phcl.p, phcl.q, reference, dx, dy)
    z, p, q, w = cg.dem, phcl.p, phcl.q, phcl.single_scattering_albedo
    all_history: list[dict[str, float]] = []
    result = None
    offset = 0
    for cycle, sigma in enumerate(sigmas, start=1):
        print(f"SfS {model.upper()} cycle {cycle}/{len(sigmas)}, r_refl={sigma:g}", flush=True)
        result = estimate_extended_sfs(
            observed, sun, view, reference,
            initial_dem=z, initial_p=p, initial_q=q, initial_albedo=w,
            pixel_size_x=dx, pixel_size_y=dy,
            parameters=warell_moon_2004(), reflectance_model=model,
            integrability_weight=2.5e-4, dem_weight=2.0e-4,
            dem_sigma_px=15.0, albedo_sigma_px=sigma,
            max_iterations=iterations_per_cycle,
            relative_tolerance=1.0e-6, iteration_offset=offset,
        )
        z, p, q, w = result.dem, result.p, result.q, result.single_scattering_albedo
        offset += len(result.history)
        leakage = albedo_topography_leakage(w, z, result.valid_mask)
        for row in result.history:
            entry = dict(row)
            entry["cycle"] = float(cycle)
            entry["albedo_topography_leakage"] = leakage
            all_history.append(entry)
    assert result is not None
    return result, all_history


def save_branch(
    directory: Path,
    name: str,
    result,
    history: list[dict[str, float]],
    observed: np.ndarray,
    reference: np.ndarray,
    template: Path,
) -> dict[str, object]:
    branch = directory / name
    branch.mkdir(parents=True, exist_ok=True)
    write_tif(branch / "final_dem.tif", result.dem, template)
    write_tif(branch / "final_minus_gld100.tif", result.dem - reference, template)
    write_tif(branch / "single_scattering_albedo.tif", result.single_scattering_albedo, template)
    write_tif(branch / "p_east.tif", result.p, template)
    write_tif(branch / "q_north.tif", result.q, template)
    for index in range(observed.shape[0]):
        write_tif(branch / f"modeled_scene_{index + 1}.tif", result.modeled_images[index], template)
        write_tif(branch / f"residual_scene_{index + 1}.tif", result.modeled_images[index] - observed[index], template)
    (branch / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    np.savez_compressed(
        branch / "result.npz", dem=result.dem, p=result.p, q_north=result.q,
        albedo=result.single_scattering_albedo, modeled=result.modeled_images,
        valid_mask=result.valid_mask,
    )
    delta = result.dem - reference
    return {
        "reflectance_fit": [image_fit(observed[i], result.modeled_images[i]) for i in range(observed.shape[0])],
        "iterations": len(history),
        "accepted_steps": int(sum(row.get("accepted", 0.0) > 0.0 for row in history)),
        "final_total_error": float(history[-1]["total_error"]) if history else float("nan"),
        "albedo_topography_leakage": albedo_topography_leakage(result.single_scattering_albedo, result.dem, result.valid_mask),
        "delta_from_gld100_rmse_m": float(np.sqrt(np.mean(delta**2))),
        "delta_from_gld100_std_m": float(np.std(delta)),
        "delta_from_gld100_min_m": float(np.min(delta)),
        "delta_from_gld100_max_m": float(np.max(delta)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--iterations-per-cycle", type=int, default=8)
    parser.add_argument(
        "--resume-sfs",
        action="store_true",
        help="Reuse successful registration and PHCL states from a prior run.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    preprocessed = root / "02_preprocessed"
    output = root / "06_complete_reproduction"
    registration_dir = output / "01_registration"
    branches_dir = output / "04_ablations"
    integration_dir = output / "05_integration"
    for path in (registration_dir, branches_dir, integration_dir):
        path.mkdir(parents=True, exist_ok=True)
    stack = np.load(preprocessed / "mairan_t_m3_stack.npz")
    images = np.asarray(stack["images"], dtype=np.float64)
    sun = np.asarray(stack["sun_directions"], dtype=np.float64)
    view = np.asarray(stack["view_directions"], dtype=np.float64)
    reference = np.asarray(stack["initial_dem"], dtype=np.float64)
    common = np.asarray(stack["common_mask"], dtype=bool)
    dx, dy = float(stack["pixel_size_x"]), float(stack["pixel_size_y"])
    scene_ids = [str(value) for value in stack["scene_ids"]]
    template = preprocessed / "GLD100_MairanT_300ppd.tif"

    if args.resume_sfs:
        saved_registration = np.load(registration_dir / "registered_m3_stack.npz")
        saved_transforms = json.loads((registration_dir / "transforms.json").read_text(encoding="utf-8"))
        registration = SimpleNamespace(
            images=np.asarray(saved_registration["images"]),
            sun_directions=np.asarray(saved_registration["sun_directions"]),
            view_directions=np.asarray(saved_registration["view_directions"]),
            descriptors=np.asarray(saved_registration["descriptors"]),
            transforms=tuple(saved_transforms[scene] for scene in scene_ids),
        )
        print("1/7 reusing completed registration", flush=True)
    else:
        print("1/7 illumination-independent subpixel registration", flush=True)
        registration = register_by_single_image_phcl(
            images, sun, view, reference,
            pixel_size_x=dx, pixel_size_y=dy,
            parameters=warell_moon_2004(), reflectance_model="amsa",
            reference_index=0, phcl_iterations=40,
        )
        (registration_dir / "transforms.json").write_text(
            json.dumps({scene_ids[i]: row for i, row in enumerate(registration.transforms)}, indent=2),
            encoding="utf-8",
        )
        np.savez_compressed(
            registration_dir / "registered_m3_stack.npz",
            images=registration.images, sun_directions=registration.sun_directions,
            view_directions=registration.view_directions, descriptors=registration.descriptors,
            initial_dem=reference, common_mask=common, scene_ids=np.asarray(scene_ids),
            pixel_size_x=dx, pixel_size_y=dy,
        )
        fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
        for i in range(3):
            axes[i, 0].imshow(images[i], cmap="gray")
            axes[i, 1].imshow(registration.descriptors[i], cmap="twilight")
            axes[i, 2].imshow(registration.images[i], cmap="gray")
            axes[i, 0].set_ylabel(scene_ids[i])
        for axis, title in zip(axes[0], ("Input", "single-image PHCL descriptor", "Registered")):
            axis.set_title(title)
        for axis in axes.flat:
            axis.set_xticks([]); axis.set_yticks([])
        fig.savefig(registration_dir / "registration_quicklook.png", dpi=180)
        plt.close(fig)

    print("2/7 exact Warell lunar Hapke parameters fixed", flush=True)
    parameter_record = {
        "source": "Warell (2004) lunar solution as adopted by Grumpe et al. (2014)",
        "single_scattering_albedo": "estimated pixel-wise",
        "dhg_b": 0.21, "dhg_c": 0.70,
        "roughness_degrees": 11.0,
        "opposition_amplitude_BS0": 3.1,
        "opposition_width_h": 0.11,
        "h_function": "Hapke 2002 level-2", "AMSA_legendre_order": 15,
    }
    (output / "02_warell_moon_parameters.json").write_text(
        json.dumps(parameter_record, indent=2), encoding="utf-8"
    )

    observed_registered, registered_thresholds = prepare_observed(registration.images, common)
    observed_unregistered, unregistered_thresholds = prepare_observed(images, common)
    if args.resume_sfs:
        def load_phcl(path: Path):
            saved = np.load(path)
            return SimpleNamespace(
                p=np.asarray(saved["p"]), q=np.asarray(saved["q_north"]),
                single_scattering_albedo=np.asarray(saved["albedo"]),
                modeled_images=np.asarray(saved["modeled"]),
                valid_mask=np.asarray(saved["valid_mask"]),
            )
        phcl_registered = load_phcl(branches_dir / "registered_amsa_multiscale" / "result.npz")
        phcl_unregistered = load_phcl(branches_dir / "unregistered_amsa_multiscale" / "result.npz")
        phcl_imsa = load_phcl(branches_dir / "registered_imsa_multiscale" / "result.npz")
        print("3/7 reusing completed PHCL states", flush=True)
    else:
        print("3/7 registered AMSA PHCL initialization", flush=True)
        phcl_registered = run_phcl(
            observed_registered, registration.sun_directions, registration.view_directions,
            reference, dx, dy, "amsa",
        )
        phcl_unregistered = run_phcl(observed_unregistered, sun, view, reference, dx, dy, "amsa")
        phcl_imsa = run_phcl(
            observed_registered, registration.sun_directions, registration.view_directions,
            reference, dx, dy, "imsa",
        )

    print("4/7 extended SfS and 5/7 multi-scale albedo", flush=True)
    multiscale = [float(value) for value in range(21, 6, -1)]
    full, full_history = run_sfs_cycles(
        observed_registered, registration.sun_directions, registration.view_directions,
        reference, dx, dy, phcl_registered, "amsa", multiscale, args.iterations_per_cycle,
    )
    unregistered, unregistered_history = run_sfs_cycles(
        observed_unregistered, sun, view, reference, dx, dy, phcl_unregistered,
        "amsa", multiscale, args.iterations_per_cycle,
    )
    imsa, imsa_history = run_sfs_cycles(
        observed_registered, registration.sun_directions, registration.view_directions,
        reference, dx, dy, phcl_imsa, "imsa", multiscale, args.iterations_per_cycle,
    )
    single_scale, single_history = run_sfs_cycles(
        observed_registered, registration.sun_directions, registration.view_directions,
        reference, dx, dy, phcl_registered, "amsa", [21.0],
        args.iterations_per_cycle * len(multiscale),
    )

    metrics = {
        "registered_amsa_multiscale": save_branch(
            branches_dir, "registered_amsa_multiscale", full, full_history,
            observed_registered, reference, template,
        ),
        "unregistered_amsa_multiscale": save_branch(
            branches_dir, "unregistered_amsa_multiscale", unregistered, unregistered_history,
            observed_unregistered, reference, template,
        ),
        "registered_imsa_multiscale": save_branch(
            branches_dir, "registered_imsa_multiscale", imsa, imsa_history,
            observed_registered, reference, template,
        ),
        "registered_amsa_single_scale": save_branch(
            branches_dir, "registered_amsa_single_scale", single_scale, single_history,
            observed_registered, reference, template,
        ),
    }

    phcl_cg = integrate_cg(phcl_registered.p, phcl_registered.q, reference, dx, dy)
    write_tif(branches_dir / "phcl_only_cg_dem.tif", phcl_cg.dem, template)
    print("6/7 literal paper relaxation versus CG", flush=True)
    literal = solve_paper_relaxation(
        full.p * dx, -full.q * dy, reference,
        pixel_size_x=1.0, pixel_size_y=1.0,
        depth_weight=100.0, lowpass_sigma_px=15.0,
        max_successful_iterations=600, max_steps_without_improvement=120,
        max_total_updates=1200, relative_tolerance=1.0e-10,
    )
    final_cg = integrate_cg(full.p, full.q, reference, dx, dy)
    write_tif(integration_dir / "paper_literal_relaxation_dem.tif", literal.dem, template)
    write_tif(integration_dir / "cg_same_objective_dem.tif", final_cg.dem, template)
    write_tif(integration_dir / "literal_minus_cg.tif", literal.dem - final_cg.dem, template)
    (integration_dir / "literal_history.json").write_text(
        json.dumps(list(literal.history), indent=2), encoding="utf-8"
    )
    metrics["integration_ablation"] = {
        "literal_stop_reason": literal.stop_reason,
        "literal_successful_iterations": literal.successful_iterations,
        "literal_total_updates": literal.total_updates,
        "literal_best_error": literal.best_error,
        "cg_converged": final_cg.converged,
        "cg_iterations": final_cg.iterations,
        "literal_minus_cg_rmse_m": float(np.sqrt(np.mean((literal.dem - final_cg.dem) ** 2))),
    }
    metrics["phcl_vs_full_sfs"] = {
        "phcl_only_cg_minus_gld100_rmse_m": float(np.sqrt(np.mean((phcl_cg.dem - reference) ** 2))),
        "full_sfs_minus_gld100_rmse_m": metrics["registered_amsa_multiscale"]["delta_from_gld100_rmse_m"],
        "phcl_reflectance_fit": [image_fit(observed_registered[i], phcl_registered.modeled_images[i]) for i in range(3)],
        "full_sfs_reflectance_fit": metrics["registered_amsa_multiscale"]["reflectance_fit"],
    }
    metrics["settings"] = {
        "scene_ids": scene_ids,
        "registered_shadow_thresholds": registered_thresholds,
        "unregistered_shadow_thresholds": unregistered_thresholds,
        "phcl_d": 1.0e-4, "phcl_r_dem_px": 15.0,
        "sfs_c": 2.5e-4, "sfs_d": 2.0e-4, "sfs_r_dem_px": 15.0,
        "albedo_sigma_sequence_px": multiscale,
        "iterations_per_cycle": args.iterations_per_cycle,
        "integration_s": 100.0, "integration_r_dem_px": 15.0,
    }
    (output / "ablation_metrics_pre_lola.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 4, figsize=(17, 9), constrained_layout=True)
    branches = [
        (reference, "GLD100"), (phcl_cg.dem, "PHCL + CG"),
        (full.dem, "registered AMSA multiscale"),
        (unregistered.dem, "unregistered AMSA multiscale"),
    ]
    for axis, (array, title) in zip(axes[0], branches):
        im = axis.imshow(array, cmap="terrain"); axis.set_title(title); axis.axis("off")
        fig.colorbar(im, ax=axis, shrink=0.7)
    differences = [
        (full.dem - reference, "full - GLD100"),
        (unregistered.dem - full.dem, "unregistered - registered"),
        (imsa.dem - full.dem, "IMSA - AMSA"),
        (single_scale.dem - full.dem, "single - multiscale"),
    ]
    for axis, (array, title) in zip(axes[1], differences):
        limit = max(float(np.percentile(np.abs(array), 99.0)), 1.0e-6)
        im = axis.imshow(array, cmap="RdBu_r", vmin=-limit, vmax=limit)
        axis.set_title(title); axis.axis("off"); fig.colorbar(im, ax=axis, shrink=0.7)
    fig.savefig(output / "seven_step_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
