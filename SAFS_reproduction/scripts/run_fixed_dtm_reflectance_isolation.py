#!/usr/bin/env python3
"""Fixed-DTM eightfold isolation of lunar reflectance laws.

The terrain, images, geometry, masks, albedo estimator, exposure estimator,
and validation pixels are identical.  Only the angular reflectance law and
its fixed operational coefficients differ.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy.stats import pearsonr, spearmanr, wilcoxon


ROOT = Path("/mnt/e/NAC_Photometry/paper2016_multi")
GRUMPE_CODE = Path("/mnt/e/光度法代码/GRUMPE方法")
SAFS_CODE = Path("/mnt/e/光度法代码/SAFS方法")
sys.path.insert(0, str(GRUMPE_CODE / "src"))
sys.path.insert(0, str(SAFS_CODE / "src"))

from grumpe_method.reflectance import hapke_amsa, hapke_imsa, warell_moon_2004  # noqa: E402
from safs_method.model import (  # noqa: E402
    direction_from_azimuth_zenith,
    normals_from_slopes,
    pixel_slopes,
)


SCENES = [
    "M1101537509RE",
    "M1149859210LE",
    "M1193400008LE",
    "M1361553043RE",
    "M1389727403LE",
    "M1504596515RE",
    "M1504610541RE",
    "M173246166LE",
]
MODELS = [
    "asp_hapke",
    "grumpe_imsa",
    "grumpe_amsa",
    "safs_lunar_lambert",
]
MODEL_LABELS = {
    "asp_hapke": "ASP Hapke",
    "grumpe_imsa": "GRUMPE Hapke IMSA",
    "grumpe_amsa": "GRUMPE Hapke AMSA",
    "safs_lunar_lambert": "SAfS Lunar-Lambert",
}

IMAGE_ROOT = ROOT / "13_grumpe_validation/03_geometry/full9/images_2m"
GEOMETRY_ROOT = ROOT / "13_grumpe_validation/03_geometry/ba_pixel_geometry_full9_direct2m"
FIXED_DTM = ROOT / "24_independent_stereo_validation/02_aligned/reference_rimasharp3_on_aligned_model_grid.tif"
FIXED_MASK = ROOT / "24_independent_stereo_validation/02_aligned/strict_common_mask.tif"
OUTPUT = ROOT / "29_fixed_dtm_reflectance_isolation"
REFLECTANCE_ROOT = OUTPUT / "01_reflectance_maps"
FOLD_ROOT = OUTPUT / "02_eightfold"
COMPARISON_ROOT = OUTPUT / "03_comparison"

ASP_COEFFS = {
    "omega": 0.68,
    "b": 0.17,
    "c": 0.62,
    "B0": 0.52,
    "h": 0.52,
}
GRUMPE_PARAMETERS = warell_moon_2004(0.4, h_function="level2", legendre_order=15)
SAFS_L = 0.35


def read(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as source:
        values = source.read(1).astype(np.float64)
        if source.nodata is not None:
            values[np.isclose(values, source.nodata)] = np.nan
        return values, source.profile.copy()


def write(path: Path, values: np.ndarray, profile: dict, byte: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = profile.copy()
    output.pop("blockxsize", None)
    output.pop("blockysize", None)
    nodata = 0 if byte else -9999.0
    output.update(
        driver="GTiff",
        count=1,
        dtype="uint8" if byte else "float32",
        nodata=nodata,
        compress="deflate",
        tiled=False,
    )
    encoded = np.where(np.isfinite(values), values, nodata)
    with rasterio.open(path, "w", **output) as target:
        target.write(encoded.astype(output["dtype"]), 1)


def read_geometry(scene: str) -> dict[str, np.ndarray]:
    folder = GEOMETRY_ROOT / scene / "maps_2m"
    suffixes = {
        "incidence": "local_incidence",
        "emission": "local_emission",
        "sun_azimuth": "sun_azimuth",
        "spacecraft_azimuth": "spacecraft_azimuth",
    }
    return {
        name: read(folder / f"{scene}_{suffix}_2m.tif")[0]
        for name, suffix in suffixes.items()
    }


def asp_hapke(mu0: np.ndarray, mu: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Exact angular formula in ASP 3.7.0 src/asp/SfS/SfsModel.cc."""
    omega = ASP_COEFFS["omega"]
    b = ASP_COEFFS["b"]
    c = ASP_COEFFS["c"]
    b0 = ASP_COEFFS["B0"]
    h = ASP_COEFFS["h"]
    cosine = np.cos(phase)
    pg = (
        (1.0 - c) * (1.0 - b * b) / np.maximum(1.0 + 2.0 * b * cosine + b * b, 1.0e-15) ** 1.5
        + c * (1.0 - b * b) / np.maximum(1.0 - 2.0 * b * cosine + b * b, 1.0e-15) ** 1.5
    )
    bg = b0 / (1.0 + np.tan(0.5 * phase) / h)
    gamma = np.sqrt(1.0 - omega)
    h0 = (1.0 + 2.0 * mu0) / np.maximum(1.0 + 2.0 * mu0 * gamma, 1.0e-15)
    hv = (1.0 + 2.0 * mu) / np.maximum(1.0 + 2.0 * mu * gamma, 1.0e-15)
    reflectance = omega / (4.0 * np.pi) * mu0 / np.maximum(mu0 + mu, 1.0e-15) * ((1.0 + bg) * pg + h0 * hv - 1.0)
    return np.where((mu0 > 0.0) & (mu > 0.0), reflectance, np.nan)


def lunar_lambert(mu0: np.ndarray, mu: np.ndarray) -> np.ndarray:
    reflectance = 2.0 * SAFS_L * mu0 / np.maximum(mu0 + mu, 1.0e-15) + (1.0 - SAFS_L) * mu0
    return np.where((mu0 > 0.0) & (mu > 0.0), reflectance, np.nan)


def scene_reflectances(
    scene: str,
    normals: np.ndarray,
    fixed_valid: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray | float]]:
    geo = read_geometry(scene)
    geometry_valid = np.ones(fixed_valid.shape, dtype=bool)
    for values in geo.values():
        geometry_valid &= np.isfinite(values)
    sun = direction_from_azimuth_zenith(geo["sun_azimuth"], geo["incidence"])
    view = direction_from_azimuth_zenith(geo["spacecraft_azimuth"], geo["emission"])
    mu0 = np.sum(normals * sun, axis=-1)
    mu = np.sum(normals * view, axis=-1)
    phase = np.arccos(np.clip(np.sum(sun * view, axis=-1), -1.0, 1.0))
    active = fixed_valid & geometry_valid & (mu0 > 0.0) & (mu > 0.0)
    reflectances = {
        "asp_hapke": asp_hapke(mu0, mu, phase),
        "grumpe_imsa": hapke_imsa(mu0, mu, phase, GRUMPE_PARAMETERS),
        "grumpe_amsa": hapke_amsa(mu0, mu, phase, GRUMPE_PARAMETERS),
        "safs_lunar_lambert": lunar_lambert(mu0, mu),
    }
    for model in reflectances:
        reflectances[model] = np.where(active & np.isfinite(reflectances[model]) & (reflectances[model] > 0.0), reflectances[model], np.nan)
    angular = {
        "incidence": np.degrees(np.arccos(np.clip(mu0, -1.0, 1.0))),
        "emission": np.degrees(np.arccos(np.clip(mu, -1.0, 1.0))),
        "phase": np.degrees(phase),
        "active": active,
    }
    return reflectances, angular


def fit_albedo_exposures(
    images: np.ndarray,
    reflectances: np.ndarray,
    valid: np.ndarray,
    iterations: int = 30,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Robust alternating fit of one shared multiplicative albedo and exposures."""
    count, rows, columns = images.shape
    exposures = np.ones(count, dtype=np.float64)
    albedo = np.ones((rows, columns), dtype=np.float64)
    history = []
    previous = exposures.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for iteration in range(iterations):
            ratios = np.where(valid, images / np.maximum(exposures[:, None, None] * reflectances, 1.0e-15), np.nan)
            albedo = np.nanmedian(ratios, axis=0)
            support = np.sum(np.isfinite(ratios), axis=0)
            albedo[support < 3] = np.nan
            finite = np.isfinite(albedo) & (albedo > 0.0)
            if finite.sum() < 1000:
                raise RuntimeError("Too few pixels to estimate shared albedo")
            low, high = np.nanpercentile(albedo[finite], (0.2, 99.8))
            albedo = np.where(finite, np.clip(albedo, low, high), np.nan)
            for index in range(count):
                basis = albedo * reflectances[index]
                active = valid[index] & np.isfinite(basis) & (basis > 0.0)
                exposures[index] = np.sum(images[index][active] * basis[active]) / np.sum(basis[active] ** 2)
            geometric_mean = float(np.exp(np.mean(np.log(np.maximum(exposures, 1.0e-15)))))
            exposures /= geometric_mean
            albedo *= geometric_mean
            change = float(np.max(np.abs(np.log(np.maximum(exposures, 1.0e-15) / np.maximum(previous, 1.0e-15)))))
            prediction = exposures[:, None, None] * albedo[None, :, :] * reflectances
            residual_values = (prediction - images)[valid & np.isfinite(prediction)]
            history.append({
                "iteration": iteration,
                "maximum_log_exposure_change": change,
                "training_rmse": float(np.sqrt(np.mean(residual_values**2))),
            })
            if change < 1.0e-7:
                break
            previous = exposures.copy()
    return albedo, exposures, history


def metrics(observed: np.ndarray, predicted: np.ndarray, valid: np.ndarray) -> dict[str, float | int]:
    residual = predicted[valid] - observed[valid]
    return {
        "pixels": int(valid.sum()),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "nrmse_median": float(np.sqrt(np.mean(residual**2)) / np.median(observed[valid])),
        "correlation": float(np.corrcoef(observed[valid], predicted[valid])[0, 1]),
        "p95_abs_residual": float(np.percentile(np.abs(residual), 95.0)),
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    COMPARISON_ROOT.mkdir(parents=True, exist_ok=True)
    dem, _ = read(FIXED_DTM)
    mask, _ = read(FIXED_MASK)
    image0, profile = read(IMAGE_ROOT / f"{SCENES[0]}.tif")
    if dem.shape != image0.shape or mask.shape != image0.shape:
        raise ValueError("Fixed DTM, mask, and common image grid must have identical array shape")
    fixed_valid = np.isfinite(dem) & np.isfinite(mask) & (mask > 0.5)
    east, north = pixel_slopes(dem, 2.0, 2.0)
    normals = normals_from_slopes(east, north)

    images = {}
    base_valid = {}
    angles = {}
    reflectances = {model: {} for model in MODELS}
    geometry_rows = []
    for scene in SCENES:
        observed, scene_profile = read(IMAGE_ROOT / f"{scene}.tif")
        if observed.shape != dem.shape or scene_profile["transform"] != profile["transform"]:
            raise ValueError(f"Image grid mismatch: {scene}")
        scene_refl, angular = scene_reflectances(scene, normals, fixed_valid)
        active = fixed_valid & np.isfinite(observed) & (observed > 0.0) & angular["active"]
        cutoff = float(np.percentile(observed[active], 1.0))
        active &= observed > cutoff
        images[scene] = observed
        base_valid[scene] = active
        angles[scene] = angular
        for model in MODELS:
            reflectances[model][scene] = scene_refl[model]
            write(REFLECTANCE_ROOT / model / f"{scene}_REFLECTANCE.tif", scene_refl[model], profile)
        geometry_rows.append({
            "scene_id": scene,
            "valid_pixels": int(active.sum()),
            "median_incidence_deg": float(np.median(angular["incidence"][active])),
            "median_emission_deg": float(np.median(angular["emission"][active])),
            "median_phase_deg": float(np.median(angular["phase"][active])),
            "median_sun_azimuth_deg": float(np.median(read_geometry(scene)["sun_azimuth"][active])),
            "median_spacecraft_azimuth_deg": float(np.median(read_geometry(scene)["spacecraft_azimuth"][active])),
        })
        print(f"REFLECTANCE_COMPLETE {scene} pixels={active.sum()}", flush=True)

    with (COMPARISON_ROOT / "scene_geometry_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(geometry_rows[0]))
        writer.writeheader()
        writer.writerows(geometry_rows)

    preliminary = {}
    fit_rows = []
    for holdout in SCENES:
        train = [scene for scene in SCENES if scene != holdout]
        image_stack = np.stack([images[scene] for scene in train])
        for model in MODELS:
            refl_stack = np.stack([reflectances[model][scene] for scene in train])
            valid_stack = np.stack([
                base_valid[scene] & np.isfinite(reflectances[model][scene])
                for scene in train
            ])
            albedo, exposures, history = fit_albedo_exposures(image_stack, refl_stack, valid_stack)
            target_refl = reflectances[model][holdout]
            basis = albedo * target_refl
            valid_target = base_valid[holdout] & np.isfinite(basis) & (basis > 0.0)
            gain = float(np.sum(images[holdout][valid_target] * basis[valid_target]) / np.sum(basis[valid_target] ** 2))
            prediction = gain * basis
            residual = np.where(valid_target, prediction - images[holdout], np.nan)
            folder = FOLD_ROOT / holdout / model
            write(folder / "albedo_from_train7.tif", albedo, profile)
            write(folder / "holdout_prediction.tif", np.where(valid_target, prediction, np.nan), profile)
            write(folder / "holdout_residual.tif", residual, profile)
            with (folder / "fit_history.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)
            metadata = {
                "holdout_id": holdout,
                "model": model,
                "model_label": MODEL_LABELS[model],
                "train_ids": train,
                "train_exposures": {scene: float(value) for scene, value in zip(train, exposures)},
                "holdout_exposure_gain": gain,
                "alternating_iterations": len(history),
                "preliminary_metrics": metrics(images[holdout], prediction, valid_target),
            }
            (folder / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            preliminary[(holdout, model)] = (prediction, residual, valid_target)
            print(f"FOLD_COMPLETE {holdout} {model} rmse={metadata['preliminary_metrics']['rmse']:.9g}", flush=True)

    strict_rows = []
    strict_masks = {}
    for holdout in SCENES:
        common = base_valid[holdout].copy()
        for model in MODELS:
            common &= preliminary[(holdout, model)][2] & np.isfinite(preliminary[(holdout, model)][1])
        strict_masks[holdout] = common
        write(FOLD_ROOT / holdout / "strict_common_mask.tif", common.astype(np.uint8), profile, byte=True)
        geom = next(row for row in geometry_rows if row["scene_id"] == holdout)
        for model in MODELS:
            prediction, residual, _ = preliminary[(holdout, model)]
            row = {
                "holdout_id": holdout,
                "model": model,
                "model_label": MODEL_LABELS[model],
                **metrics(images[holdout], prediction, common),
                "median_incidence_deg": geom["median_incidence_deg"],
                "median_emission_deg": geom["median_emission_deg"],
                "median_phase_deg": geom["median_phase_deg"],
            }
            strict_rows.append(row)
            write(FOLD_ROOT / holdout / model / "strict_holdout_residual.tif", np.where(common, residual, np.nan), profile)

    with (COMPARISON_ROOT / "strict_eightfold_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(strict_rows[0]))
        writer.writeheader()
        writer.writerows(strict_rows)

    summaries = []
    for model in MODELS:
        subset = [row for row in strict_rows if row["model"] == model]
        summaries.append({
            "model": model,
            "model_label": MODEL_LABELS[model],
            "folds": len(subset),
            "rmse_mean": float(np.mean([row["rmse"] for row in subset])),
            "rmse_median": float(np.median([row["rmse"] for row in subset])),
            "rmse_std": float(np.std([row["rmse"] for row in subset], ddof=1)),
            "correlation_mean": float(np.mean([row["correlation"] for row in subset])),
            "nrmse_mean": float(np.mean([row["nrmse_median"] for row in subset])),
            "bias_abs_mean": float(np.mean([abs(row["bias"]) for row in subset])),
        })
    winner_counts = {model: 0 for model in MODELS}
    for holdout in SCENES:
        candidates = [row for row in strict_rows if row["holdout_id"] == holdout]
        winner_counts[min(candidates, key=lambda row: row["rmse"])["model"]] += 1
    best_model = min(summaries, key=lambda row: row["rmse_mean"])["model"]

    pairwise = []
    best_values = np.asarray([next(row["rmse"] for row in strict_rows if row["holdout_id"] == scene and row["model"] == best_model) for scene in SCENES])
    for model in MODELS:
        values = np.asarray([next(row["rmse"] for row in strict_rows if row["holdout_id"] == scene and row["model"] == model) for scene in SCENES])
        if model == best_model or np.allclose(values, best_values):
            statistic, pvalue = 0.0, 1.0
        else:
            statistic, pvalue = wilcoxon(values, best_values, alternative="two-sided")
        pairwise.append({
            "model": model,
            "against_best": best_model,
            "mean_rmse_difference": float(np.mean(values - best_values)),
            "mean_relative_difference_percent": float(100.0 * np.mean((values - best_values) / best_values)),
            "wilcoxon_statistic": float(statistic),
            "wilcoxon_pvalue": float(pvalue),
        })

    angle_relations = []
    for model in MODELS:
        subset = [next(row for row in strict_rows if row["holdout_id"] == scene and row["model"] == model) for scene in SCENES]
        rmse = np.asarray([row["rmse"] for row in subset])
        for field in ["median_incidence_deg", "median_emission_deg", "median_phase_deg"]:
            values = np.asarray([row[field] for row in subset])
            pearson = pearsonr(values, rmse)
            spearman = spearmanr(values, rmse)
            angle_relations.append({
                "model": model,
                "angle": field,
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
            })

    summary = {
        "protocol": "Fixed official stereo DTM, clean8 leave-one-out, common multiplicative albedo/exposure estimator, strict common pixels; only angular reflectance law differs.",
        "fixed_dtm": str(FIXED_DTM),
        "fixed_mask": str(FIXED_MASK),
        "models": {
            "asp_hapke": {"source": "ASP 3.7.0 SfsModel.cc HapkeReflectance", **ASP_COEFFS},
            "grumpe_imsa": {"source": "Grumpe reproduction Warell Moon 2004", "single_scattering_albedo": 0.4, "roughness_degrees": 11.0},
            "grumpe_amsa": {"source": "Grumpe reproduction Warell Moon 2004", "single_scattering_albedo": 0.4, "roughness_degrees": 11.0, "legendre_order": 15},
            "safs_lunar_lambert": {"source": "Wu 2016 reproduction", "L": SAFS_L},
        },
        "material_parameter_protocol": "All laws use the same fitted shared multiplicative albedo map. Hapke material coefficients are fixed; this isolates angular behavior rather than model-specific nonlinear material inversion.",
        "model_summary": summaries,
        "winner_counts": winner_counts,
        "best_mean_rmse_model": best_model,
        "pairwise_against_best": pairwise,
        "angle_relations": angle_relations,
    }
    (COMPARISON_ROOT / "reflectance_isolation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for filename, rows in [("model_summary.csv", summaries), ("pairwise_against_best.csv", pairwise), ("angle_relations.csv", angle_relations)]:
        with (COMPARISON_ROOT / filename).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    x = np.arange(len(SCENES))
    fig, axis = plt.subplots(figsize=(13, 6), constrained_layout=True)
    for model in MODELS:
        values = [next(row["rmse"] for row in strict_rows if row["holdout_id"] == scene and row["model"] == model) for scene in SCENES]
        axis.plot(x, values, marker="o", label=MODEL_LABELS[model])
    axis.set_xticks(x, SCENES, rotation=35, ha="right", fontsize=8)
    axis.set(ylabel="Strict-common holdout I/F RMSE", title="Fixed-DTM reflectance-law isolation: clean8 leave-one-out")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.savefig(COMPARISON_ROOT / "eightfold_rmse_by_model.png", dpi=190)
    plt.close(fig)

    ordered = sorted(summaries, key=lambda row: row["rmse_mean"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    axes[0].barh([row["model_label"] for row in ordered], [row["rmse_mean"] for row in ordered], xerr=[row["rmse_std"] for row in ordered])
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Mean holdout I/F RMSE ± fold SD", title="Angular reflectance accuracy")
    axes[0].grid(axis="x", alpha=0.25)
    axes[1].barh([row["model_label"] for row in ordered], [row["correlation_mean"] for row in ordered])
    axes[1].invert_yaxis()
    axes[1].set(xlabel="Mean holdout correlation", title="Prediction correlation")
    axes[1].grid(axis="x", alpha=0.25)
    fig.savefig(COMPARISON_ROOT / "model_summary.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    angle_fields = [
        ("median_incidence_deg", "Median local incidence (deg)"),
        ("median_emission_deg", "Median local emission (deg)"),
        ("median_phase_deg", "Median phase angle (deg)"),
    ]
    for axis, (field, label) in zip(axes, angle_fields):
        for model in MODELS:
            subset = [next(row for row in strict_rows if row["holdout_id"] == scene and row["model"] == model) for scene in SCENES]
            axis.plot([row[field] for row in subset], [row["rmse"] for row in subset], marker="o", linestyle="none", label=MODEL_LABELS[model])
        axis.set(xlabel=label, ylabel="Holdout I/F RMSE")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    fig.suptitle("Reflectance-law error versus illumination/view geometry (n=8 scenes)")
    fig.savefig(COMPARISON_ROOT / "error_vs_angles.png", dpi=190)
    plt.close(fig)

    (OUTPUT / "EXPERIMENT_COMPLETE.txt").write_text("FIXED_DTM_REFLECTANCE_ISOLATION_COMPLETE\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("FIXED_DTM_REFLECTANCE_ISOLATION_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
