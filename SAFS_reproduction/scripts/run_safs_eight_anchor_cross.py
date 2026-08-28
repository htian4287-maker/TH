#!/usr/bin/env python3
"""Eight-anchor SAfS cross-illumination experiment on the clean8 NAC set.

Each scene independently supplies one SAfS DEM/albedo model.  That fixed model
then predicts the other seven scenes with only one scalar exposure gain per
target.  ASP Hapke and GRUMPE AMSA leave-one-out residuals are recomputed on
the exact same per-target pixel intersection as all seven SAfS predictions.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import binary_erosion, gaussian_filter


ROOT = Path("/mnt/e/NAC_Photometry/paper2016_multi")
CODE = Path("/mnt/e/光度法代码/SAFS方法")
sys.path.insert(0, str(CODE / "src"))

from safs_method import SafsConfig, direction_from_azimuth_zenith, solve_safs  # noqa: E402
from safs_method.model import render_reflectance, resize_bilinear  # noqa: E402


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
IMAGE_ROOT = ROOT / "13_grumpe_validation/03_geometry/full9/images_2m"
GEOMETRY_ROOT = ROOT / "13_grumpe_validation/03_geometry/ba_pixel_geometry_full9_direct2m"
COARSE_DEM = ROOT.parent / "paper2016/05_safs_inputs/sldem_coarse_12x11.tif"
REFERENCE = ROOT / "24_independent_stereo_validation/02_aligned/reference_rimasharp3_on_aligned_model_grid.tif"
REFERENCE_MASK = ROOT / "24_independent_stereo_validation/02_aligned/strict_common_mask.tif"
REFERENCE_CONFIDENCE = ROOT / "24_independent_stereo_validation/02_aligned/reference_confidence_on_aligned_model_grid.tif"
OTHER_RESIDUALS = ROOT / "23_asp_hapke_clean8_eightfold/05_comparison/residuals"
MODEL_GRID = ROOT / "07_sfs_multi/03_pilot_dem/sldem_pilot_2m.tif"
MULTI_GEOMETRY_MODELS = {
    "Initial SLDEM": MODEL_GRID,
    "ASP multi-7 baseline": ROOT / "23_asp_hapke_clean8_eightfold/folds/M173246166LE/02_sfs/run-DEM-final.tif",
    "GRUMPE multi-7 AMSA": ROOT / "20_grumpe_clean8_eightfold/folds/M173246166LE/amsa/reconstructed_dem.tif",
    "ASP multi-7 strong initial": ROOT / "25_asp_regularization_m173_holdout/configs/sw0p04_iw0p001/01_sfs/run-DEM-final.tif",
}
OUTPUT = ROOT / "28_safs_eight_anchor_cross"
MODEL_ROOT = OUTPUT / "01_models"
PREDICTION_ROOT = OUTPUT / "02_cross_predictions"
COMPARISON_ROOT = OUTPUT / "03_comparison"


CONFIG = SafsConfig(
    iterations_per_level=5,
    maximum_pyramid_levels=9,
    albedo_window_final_px=61,
    reflectance_smoothing_final_px=5,
    lunar_lambert_l=0.35,
    reflectance_weight=1.0,
    normal_weight=1.0,
    finite_difference_step_m=0.05,
    maximum_height_step_m=0.25,
    newton_damping=1.0e-6,
    line_search_steps=6,
    convergence_tolerance_m=1.0e-3,
    shadow_threshold_normalized=0.08,
    shadow_percentile=1.0,
    albedo_minimum=0.35,
    albedo_maximum=2.5,
    preserve_vertical_datum=True,
    sweep_mode="four_color",
    sequential_max_pixels=65536,
)


def read(path: Path, band: int = 1) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as source:
        values = source.read(band).astype(np.float64)
        if source.nodata is not None:
            values[np.isclose(values, source.nodata)] = np.nan
        return values, source.profile.copy()


def on_profile(path: Path, profile: dict, nearest: bool = False) -> np.ndarray:
    destination = np.full((profile["height"], profile["width"]), np.nan, dtype=np.float64)
    with rasterio.open(path) as source:
        reproject(
            source=rasterio.band(source, 1),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=profile["transform"],
            dst_crs=profile["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.nearest if nearest else Resampling.bilinear,
        )
    return destination


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


def geometry(scene: str) -> dict[str, np.ndarray]:
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


def directions(scene: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = geometry(scene)
    valid = np.ones(values["incidence"].shape, dtype=bool)
    for item in values.values():
        valid &= np.isfinite(item)
    sun = direction_from_azimuth_zenith(values["sun_azimuth"], values["incidence"])
    view = direction_from_azimuth_zenith(values["spacecraft_azimuth"], values["emission"])
    return sun, view, valid


def photo_stats(observed: np.ndarray, predicted: np.ndarray, valid: np.ndarray) -> dict[str, float | int]:
    residual = predicted[valid] - observed[valid]
    return {
        "valid_pixels": int(valid.sum()),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "nrmse_median": float(np.sqrt(np.mean(residual**2)) / np.median(observed[valid])),
        "correlation": float(np.corrcoef(observed[valid], predicted[valid])[0, 1]),
        "p95_abs_residual": float(np.percentile(np.abs(residual), 95.0)),
    }


def residual_stats(observed: np.ndarray, residual: np.ndarray, valid: np.ndarray) -> dict[str, float | int]:
    predicted = observed[valid] + residual[valid]
    values = residual[valid]
    return {
        "common_pixels": int(valid.sum()),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "mae": float(np.mean(np.abs(values))),
        "bias": float(np.mean(values)),
        "nrmse_median": float(np.sqrt(np.mean(values**2)) / np.median(observed[valid])),
        "correlation": float(np.corrcoef(observed[valid], predicted)[0, 1]),
        "p95_abs_residual": float(np.percentile(np.abs(values), 95.0)),
    }


def smooth_slope(dem: np.ndarray, valid: np.ndarray, pixel_size: float = 2.0) -> np.ndarray:
    sigma = 10.0 / pixel_size
    weights = gaussian_filter(valid.astype(np.float64), sigma, mode="nearest")
    smooth = gaussian_filter(np.where(valid, dem, 0.0), sigma, mode="nearest") / np.maximum(weights, 1.0e-12)
    row, column = np.gradient(np.where(weights > 0.2, smooth, 0.0), pixel_size, pixel_size)
    slope = np.degrees(np.arctan(np.hypot(row, column)))
    slope[weights <= 0.2] = np.nan
    return slope


def build_model(scene: str, coarse: np.ndarray) -> dict:
    folder = MODEL_ROOT / scene
    metrics_path = folder / "run-METRICS.json"
    required = [folder / "run-DEM.tif", folder / "run-ALBEDO.tif", metrics_path]
    if all(path.exists() for path in required):
        print(f"REUSE_MODEL {scene}", flush=True)
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    observed, profile = read(IMAGE_ROOT / f"{scene}.tif")
    sun, view, geometry_valid = directions(scene)
    valid = np.isfinite(observed) & (observed > 0.0) & geometry_valid
    scale = float(np.median(observed[valid]))
    normalized = observed / scale

    def progress(row: dict[str, float]) -> None:
        print(
            f"MODEL {scene} level={int(row['level'])} shape={int(row['rows'])}x{int(row['columns'])} "
            f"iter={int(row['iteration'])} cost={row['cost_before']:.6g}->{row['cost_after']:.6g} "
            f"step={row['maximum_height_step_m']:.5g}",
            flush=True,
        )

    result = solve_safs(
        normalized,
        coarse,
        2.0,
        2.0,
        sun,
        view,
        config=CONFIG,
        valid_mask=valid,
        progress=progress,
    )
    initial = resize_bilinear(coarse, observed.shape)
    modeled = result.modeled_intensity * scale
    photo_valid = result.valid_mask & valid & np.isfinite(modeled)
    photo = photo_stats(observed, modeled, photo_valid)
    metrics = {
        "anchor_id": scene,
        "shape": list(observed.shape),
        "pixel_size_m": 2.0,
        "input_median_iof": scale,
        "pyramid_shapes": [list(shape) for shape in result.pyramid_shapes],
        "self_photo_rmse": photo["rmse"],
        "self_photo_correlation": photo["correlation"],
        "self_photo_pixels": photo["valid_pixels"],
        "dem_update_rmse_m": float(np.sqrt(np.nanmean((result.dem[valid] - initial[valid]) ** 2))),
        "dem_update_p95_abs_m": float(np.nanpercentile(np.abs(result.dem[valid] - initial[valid]), 95.0)),
        "albedo_mean": float(np.nanmean(result.albedo[valid])),
        "albedo_std": float(np.nanstd(result.albedo[valid])),
    }
    write(folder / "run-DEM.tif", result.dem, profile)
    write(folder / "run-ALBEDO.tif", result.albedo, profile)
    write(folder / "run-MODELED-IOF.tif", modeled, profile)
    write(folder / "run-PHOTOMETRIC-RESIDUAL.tif", np.where(photo_valid, modeled - observed, np.nan), profile)
    write(folder / "run-DEM-MINUS-INITIAL.tif", result.dem - initial, profile)
    write(folder / "run-VALID-MASK.tif", valid.astype(np.uint8), profile, byte=True)
    with (folder / "run-HISTORY.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result.history[0]))
        writer.writeheader()
        writer.writerows(result.history)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    update = result.dem - initial
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    panels = [
        ("Observed I/F", observed, "gray"),
        ("Modeled I/F", modeled, "gray"),
        ("Residual", np.where(photo_valid, modeled - observed, np.nan), "RdBu_r"),
        ("DEM update (m)", update, "RdBu_r"),
        ("Albedo", result.albedo, "gray"),
        ("Refined DEM", result.dem, "terrain"),
    ]
    for axis, (title, values, cmap) in zip(axes.flat, panels):
        good = np.isfinite(values) & valid
        if cmap == "RdBu_r":
            limit = max(float(np.nanpercentile(np.abs(values[good]), 99)), 1.0e-9)
            vmin, vmax = -limit, limit
        else:
            vmin, vmax = np.nanpercentile(values[good], (1, 99))
        artist = axis.imshow(np.where(valid, values, np.nan), cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")
        fig.colorbar(artist, ax=axis, shrink=0.72)
    fig.suptitle(f"SAfS single anchor: {scene}")
    fig.savefig(folder / "run-OVERVIEW.png", dpi=165)
    plt.close(fig)
    print(f"MODEL_COMPLETE {scene} self_rmse={metrics['self_photo_rmse']:.9g}", flush=True)
    return metrics


def cross_predict(anchor: str, target: str) -> dict:
    folder = PREDICTION_ROOT / anchor
    row_path = folder / f"{target}_METRICS.json"
    residual_path = folder / f"{target}_RESIDUAL.tif"
    prediction_path = folder / f"{target}_PREDICTION.tif"
    if row_path.exists() and residual_path.exists() and prediction_path.exists():
        print(f"REUSE_PREDICTION {anchor}->{target}", flush=True)
        return json.loads(row_path.read_text(encoding="utf-8"))

    observed, profile = read(IMAGE_ROOT / f"{target}.tif")
    dem = on_profile(MODEL_ROOT / anchor / "run-DEM.tif", profile)
    albedo = on_profile(MODEL_ROOT / anchor / "run-ALBEDO.tif", profile)
    sun, view, geometry_valid = directions(target)
    reflectance, illuminated = render_reflectance(dem, 2.0, 2.0, sun, view, CONFIG.lunar_lambert_l)
    base = albedo * reflectance
    valid = (
        np.isfinite(observed)
        & (observed > 0.0)
        & np.isfinite(base)
        & (base > 0.0)
        & geometry_valid
        & illuminated
    )
    shadow_cutoff = float(np.percentile(observed[valid], 1.0))
    valid &= observed > shadow_cutoff
    gain = float(np.sum(observed[valid] * base[valid]) / np.sum(base[valid] ** 2))
    predicted = gain * base
    row = {
        "anchor_id": anchor,
        "holdout_id": target,
        "gain": gain,
        **photo_stats(observed, predicted, valid),
    }
    folder.mkdir(parents=True, exist_ok=True)
    write(prediction_path, np.where(valid, predicted, np.nan), profile)
    write(residual_path, np.where(valid, predicted - observed, np.nan), profile)
    row_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PREDICTION_COMPLETE {anchor}->{target} rmse={row['rmse']:.9g}", flush=True)
    return row


def strict_comparison() -> tuple[list[dict], dict]:
    rows: list[dict] = []
    for target in SCENES:
        observed, profile = read(IMAGE_ROOT / f"{target}.tif")
        anchors = [scene for scene in SCENES if scene != target]
        residuals = {
            f"SAfS single {anchor}": on_profile(PREDICTION_ROOT / anchor / f"{target}_RESIDUAL.tif", profile)
            for anchor in anchors
        }
        residuals["ASP Hapke multi-7"] = on_profile(
            OTHER_RESIDUALS / f"{target}_asp_hapke_residual.tif", profile
        )
        residuals["GRUMPE AMSA multi-7"] = on_profile(
            OTHER_RESIDUALS / f"{target}_grumpe_amsa_residual.tif", profile
        )
        common = np.isfinite(observed) & (observed > 0.0)
        for residual in residuals.values():
            common &= np.isfinite(residual)
        if common.sum() < 1000:
            raise RuntimeError(f"Too few strict common pixels for {target}: {common.sum()}")
        for model, residual in residuals.items():
            is_safs = model.startswith("SAfS single ")
            rows.append(
                {
                    "holdout_id": target,
                    "model_family": "SAfS single" if is_safs else model,
                    "anchor_id": model.removeprefix("SAfS single ") if is_safs else "seven scenes excluding holdout",
                    "training_images": 1 if is_safs else 7,
                    **residual_stats(observed, residual, common),
                }
            )

    COMPARISON_ROOT.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (COMPARISON_ROOT / "strict_common_metrics_long.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    families = ["ASP Hapke multi-7", "GRUMPE AMSA multi-7", "SAfS single"]
    family_summary = []
    for family in families:
        subset = [row for row in rows if row["model_family"] == family]
        family_summary.append(
            {
                "model_family": family,
                "predictions": len(subset),
                "rmse_mean": float(np.mean([row["rmse"] for row in subset])),
                "rmse_median": float(np.median([row["rmse"] for row in subset])),
                "rmse_min": float(np.min([row["rmse"] for row in subset])),
                "rmse_max": float(np.max([row["rmse"] for row in subset])),
                "correlation_mean": float(np.mean([row["correlation"] for row in subset])),
                "nrmse_mean": float(np.mean([row["nrmse_median"] for row in subset])),
            }
        )

    anchor_summary = []
    for anchor in SCENES:
        subset = [row for row in rows if row["model_family"] == "SAfS single" and row["anchor_id"] == anchor]
        anchor_summary.append(
            {
                "anchor_id": anchor,
                "holdouts": len(subset),
                "rmse_mean": float(np.mean([row["rmse"] for row in subset])),
                "rmse_median": float(np.median([row["rmse"] for row in subset])),
                "correlation_mean": float(np.mean([row["correlation"] for row in subset])),
                "beats_asp": int(sum(row["rmse"] < next(item["rmse"] for item in rows if item["holdout_id"] == row["holdout_id"] and item["model_family"] == "ASP Hapke multi-7") for row in subset)),
                "beats_grumpe": int(sum(row["rmse"] < next(item["rmse"] for item in rows if item["holdout_id"] == row["holdout_id"] and item["model_family"] == "GRUMPE AMSA multi-7") for row in subset)),
            }
        )

    holdout_summary = []
    for target in SCENES:
        safs = [row for row in rows if row["holdout_id"] == target and row["model_family"] == "SAfS single"]
        asp = next(row for row in rows if row["holdout_id"] == target and row["model_family"] == "ASP Hapke multi-7")
        grumpe = next(row for row in rows if row["holdout_id"] == target and row["model_family"] == "GRUMPE AMSA multi-7")
        best = min(safs, key=lambda row: row["rmse"])
        holdout_summary.append(
            {
                "holdout_id": target,
                "common_pixels": asp["common_pixels"],
                "asp_rmse": asp["rmse"],
                "grumpe_rmse": grumpe["rmse"],
                "safs_mean_rmse": float(np.mean([row["rmse"] for row in safs])),
                "safs_median_rmse": float(np.median([row["rmse"] for row in safs])),
                "safs_best_rmse": best["rmse"],
                "safs_best_anchor": best["anchor_id"],
            }
        )

    summary = {
        "protocol": "clean8: each SAfS model is trained on one anchor and predicts the other seven; all seven SAfS residuals plus ASP/GRUMPE residuals share one strict pixel intersection per holdout",
        "scenes": SCENES,
        "safs_cross_predictions": 56,
        "family_summary": family_summary,
        "anchor_summary": anchor_summary,
        "holdout_summary": holdout_summary,
    }
    (COMPARISON_ROOT / "strict_common_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name, values in [("anchor_summary.csv", anchor_summary), ("holdout_summary.csv", holdout_summary)]:
        with (COMPARISON_ROOT / name).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    return rows, summary


def geometry_comparison() -> tuple[list[dict], list[dict]]:
    # The independent stereo alignment determined one shared horizontal shift
    # (east=-34 m, north=-30 m).  Every model remains sampled on the original
    # 2 m grid, then its values are compared by index with the shifted reference
    # grid.  Reprojecting the reference back to the unshifted transform would
    # apply the displacement twice and create a false pixel phase.
    reference, reference_profile = read(REFERENCE)
    reference_mask, _ = read(REFERENCE_MASK)
    confidence, _ = read(REFERENCE_CONFIDENCE)
    _, model_profile = read(MODEL_GRID)
    models = {f"SAfS single {scene}": read(MODEL_ROOT / scene / "run-DEM.tif")[0] for scene in SCENES}
    for name, path in MULTI_GEOMETRY_MODELS.items():
        models[name] = on_profile(path, model_profile)
    if any(values.shape != reference.shape for values in models.values()):
        raise ValueError("SAfS model and shifted stereo reference shapes differ")
    common = np.isfinite(reference) & (reference_mask > 0.5)
    for values in models.values():
        common &= np.isfinite(values)
    fit_mask = np.isfinite(reference) & np.isfinite(confidence) & (confidence >= 10) & (confidence <= 15)
    for values in models.values():
        fit_mask &= np.isfinite(values)
    fit_mask = binary_erosion(fit_mask, iterations=3)
    rows = []
    reference_slope = smooth_slope(reference, common)
    for name, dem in models.items():
        dz = float(np.median(reference[fit_mask] - dem[fit_mask]))
        aligned = dem + dz
        elevation = aligned - reference
        slope = smooth_slope(aligned, common)
        slope_valid = common & np.isfinite(reference_slope) & np.isfinite(slope)
        is_safs = name.startswith("SAfS single ")
        if is_safs:
            family = "SAfS single"
            anchor = name.removeprefix("SAfS single ")
            training_images = 1
            uses_m173 = anchor == "M173246166LE"
        elif name == "Initial SLDEM":
            family = "Initial"
            anchor = ""
            training_images = 0
            uses_m173 = False
        elif name.startswith("GRUMPE"):
            family = "GRUMPE multi-7"
            anchor = "seven scenes excluding M173"
            training_images = 7
            uses_m173 = False
        else:
            family = "ASP multi-7"
            anchor = "seven scenes excluding M173"
            training_images = 7
            uses_m173 = False
        rows.append(
            {
                "model": name,
                "model_family": family,
                "anchor_id": anchor,
                "training_images": training_images,
                "uses_M173_in_optimization": uses_m173,
                "common_pixels": int(common.sum()),
                "vertical_shift_m": dz,
                "elevation_rmse_m": float(np.sqrt(np.mean(elevation[common] ** 2))),
                "elevation_mae_m": float(np.mean(np.abs(elevation[common]))),
                "slope20_rmse_deg": float(np.sqrt(np.mean((slope[slope_valid] - reference_slope[slope_valid]) ** 2))),
                "slope20_mae_deg": float(np.mean(np.abs(slope[slope_valid] - reference_slope[slope_valid]))),
            }
        )
        safe_name = name.replace(" ", "_").replace("/", "_")
        write(COMPARISON_ROOT / "geometry" / f"{safe_name}_ELEVATION_MINUS_STEREO.tif", np.where(common, elevation, np.nan), reference_profile)
    with (COMPARISON_ROOT / "all_geometry_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    safs_rows = [row for row in rows if row["model_family"] == "SAfS single"]
    with (COMPARISON_ROOT / "safs_anchor_geometry_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(safs_rows[0]))
        writer.writeheader()
        writer.writerows(safs_rows)
    return safs_rows, rows


def figures(rows: list[dict], summary: dict, geometry_rows: list[dict], all_geometry_rows: list[dict]) -> None:
    matrix = np.full((len(SCENES), len(SCENES)), np.nan)
    for row in rows:
        if row["model_family"] == "SAfS single":
            matrix[SCENES.index(row["anchor_id"]), SCENES.index(row["holdout_id"])] = row["rmse"]
    fig, axis = plt.subplots(figsize=(10.5, 8), constrained_layout=True)
    artist = axis.imshow(matrix, cmap="viridis")
    axis.set_xticks(range(len(SCENES)), SCENES, rotation=40, ha="right", fontsize=8)
    axis.set_yticks(range(len(SCENES)), SCENES, fontsize=8)
    axis.set_xlabel("Unseen illumination / holdout")
    axis.set_ylabel("Single-image SAfS anchor")
    axis.set_title("SAfS 8-anchor cross-illumination I/F RMSE")
    for row_index in range(len(SCENES)):
        for column_index in range(len(SCENES)):
            value = matrix[row_index, column_index]
            if np.isfinite(value):
                axis.text(column_index, row_index, f"{value:.4f}", ha="center", va="center", fontsize=6, color="white" if value > np.nanmedian(matrix) else "black")
    fig.colorbar(artist, ax=axis, label="Strict-common I/F RMSE")
    fig.savefig(COMPARISON_ROOT / "safs_8x8_cross_rmse_heatmap.png", dpi=190)
    plt.close(fig)

    holdouts = summary["holdout_summary"]
    x = np.arange(len(SCENES))
    fig, axis = plt.subplots(figsize=(13, 6), constrained_layout=True)
    axis.plot(x, [row["asp_rmse"] for row in holdouts], marker="o", label="ASP Hapke multi-7")
    axis.plot(x, [row["grumpe_rmse"] for row in holdouts], marker="o", label="GRUMPE AMSA multi-7")
    axis.plot(x, [row["safs_mean_rmse"] for row in holdouts], marker="o", label="SAfS single: mean of 7 anchors")
    axis.plot(x, [row["safs_best_rmse"] for row in holdouts], marker="o", linestyle="--", label="SAfS single: best anchor (optimistic)")
    axis.set_xticks(x, SCENES, rotation=35, ha="right", fontsize=8)
    axis.set_ylabel("I/F RMSE on strict common pixels")
    axis.set_title("Single-image SAfS vs multi-image models across all clean8 holdouts")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.savefig(COMPARISON_ROOT / "single_vs_multi_all_holdouts.png", dpi=190)
    plt.close(fig)

    anchors = sorted(summary["anchor_summary"], key=lambda row: row["rmse_mean"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    axes[0].barh([row["anchor_id"] for row in anchors], [row["rmse_mean"] for row in anchors])
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Mean unseen-light I/F RMSE", title="SAfS anchor generalization ranking")
    axes[0].grid(axis="x", alpha=0.25)
    ordered_geometry = sorted(geometry_rows, key=lambda row: row["elevation_rmse_m"])
    axes[1].barh([row["anchor_id"] for row in ordered_geometry], [row["elevation_rmse_m"] for row in ordered_geometry])
    axes[1].invert_yaxis()
    axes[1].set(xlabel="Elevation RMSE against stereo DTM (m)", title="SAfS anchor geometry ranking")
    axes[1].grid(axis="x", alpha=0.25)
    fig.savefig(COMPARISON_ROOT / "safs_anchor_rankings.png", dpi=190)
    plt.close(fig)

    selected = [
        next(row for row in all_geometry_rows if row["model"] == "Initial SLDEM"),
        next(row for row in all_geometry_rows if row["model"] == "ASP multi-7 baseline"),
        next(row for row in all_geometry_rows if row["model"] == "GRUMPE multi-7 AMSA"),
        next(row for row in all_geometry_rows if row["model"] == "ASP multi-7 strong initial"),
        min((row for row in geometry_rows if row["anchor_id"] != "M173246166LE"), key=lambda row: row["elevation_rmse_m"]),
        next(row for row in geometry_rows if row["anchor_id"] == "M173246166LE"),
    ]
    labels = [row["model"] if row["model_family"] != "SAfS single" else f"SAfS {row['anchor_id']}" for row in selected]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    axes[0].barh(labels, [row["elevation_rmse_m"] for row in selected])
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Elevation RMSE (m)", title="Common-mask geometry: single vs multi")
    axes[0].grid(axis="x", alpha=0.25)
    axes[1].barh(labels, [row["slope20_rmse_deg"] for row in selected])
    axes[1].invert_yaxis()
    axes[1].set(xlabel="20 m slope RMSE (degree)", title="Common-mask slope: single vs multi")
    axes[1].grid(axis="x", alpha=0.25)
    fig.savefig(COMPARISON_ROOT / "single_vs_multi_geometry_common.png", dpi=190)
    plt.close(fig)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    COMPARISON_ROOT.mkdir(parents=True, exist_ok=True)
    coarse, _ = read(COARSE_DEM)
    model_metrics = [build_model(scene, coarse) for scene in SCENES]
    with (COMPARISON_ROOT / "model_self_fit_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(model_metrics[0]))
        writer.writeheader()
        writer.writerows(model_metrics)

    raw_predictions = []
    for anchor in SCENES:
        for target in SCENES:
            if anchor != target:
                raw_predictions.append(cross_predict(anchor, target))
    with (COMPARISON_ROOT / "raw_cross_prediction_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(raw_predictions[0]))
        writer.writeheader()
        writer.writerows(raw_predictions)

    rows, summary = strict_comparison()
    geometry_rows, all_geometry_rows = geometry_comparison()
    figures(rows, summary, geometry_rows, all_geometry_rows)
    (OUTPUT / "EXPERIMENT_COMPLETE.txt").write_text(
        "SAFS_EIGHT_ANCHOR_CROSS_COMPLETE\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("SAFS_EIGHT_ANCHOR_CROSS_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
