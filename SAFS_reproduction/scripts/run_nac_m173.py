#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from scipy.ndimage import gaussian_filter
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from safs_method import SafsConfig, direction_from_azimuth_zenith, solve_safs  # noqa: E402
from safs_method.model import resize_bilinear  # noqa: E402


REFERENCE = Path("/mnt/e/NAC_Photometry/paper2016_multi/24_independent_stereo_validation/02_aligned/reference_rimasharp3_on_aligned_model_grid.tif")
REFERENCE_MASK = Path("/mnt/e/NAC_Photometry/paper2016_multi/24_independent_stereo_validation/02_aligned/strict_common_mask.tif")


def load_document(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def read_resampled(path: Path, band: int, shape: tuple[int, int], resampling: Resampling) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as source:
        values = source.read(band, out_shape=shape, resampling=resampling).astype(np.float64)
        if source.nodata is not None:
            values[np.isclose(values, source.nodata)] = np.nan
        scale_x = source.width / shape[1]
        scale_y = source.height / shape[0]
        profile = source.profile.copy()
        profile.update(
            height=shape[0], width=shape[1], count=1,
            transform=source.transform * source.transform.scale(scale_x, scale_y),
        )
        return values, profile


def write_raster(path: Path, values: np.ndarray, profile: dict, *, byte: bool = False) -> None:
    output = profile.copy()
    nodata = 0 if byte else -9999.0
    output.update(
        driver="GTiff", count=1, dtype="uint8" if byte else "float32",
        nodata=nodata, compress="deflate", tiled=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = np.where(np.isfinite(values), values, nodata)
    with rasterio.open(path, "w", **output) as target:
        target.write(encoded.astype(output["dtype"]), 1)


def reproject_reference(path: Path, profile: dict, nearest: bool = False) -> np.ndarray:
    destination = np.full((profile["height"], profile["width"]), np.nan, dtype=np.float64)
    with rasterio.open(path) as source:
        reproject(
            source=rasterio.band(source, 1), destination=destination,
            src_transform=source.transform, src_crs=source.crs, src_nodata=source.nodata,
            dst_transform=profile["transform"], dst_crs=profile["crs"], dst_nodata=np.nan,
            resampling=Resampling.nearest if nearest else Resampling.bilinear,
        )
    return destination


def error_stats(residual: np.ndarray, valid: np.ndarray, prefix: str) -> dict:
    values = residual[valid]
    median = float(np.median(values))
    return {
        f"{prefix}_pixels": int(values.size),
        f"{prefix}_rmse": float(np.sqrt(np.mean(values**2))),
        f"{prefix}_mae": float(np.mean(np.abs(values))),
        f"{prefix}_bias": float(np.mean(values)),
        f"{prefix}_nmad": float(1.4826 * np.median(np.abs(values - median))),
        f"{prefix}_p95_abs": float(np.percentile(np.abs(values), 95.0)),
    }


def smooth_slope(dem: np.ndarray, valid: np.ndarray, pixel_size: float) -> np.ndarray:
    sigma = 10.0 / pixel_size
    weight = gaussian_filter(valid.astype(np.float64), sigma, mode="nearest")
    smooth = gaussian_filter(np.where(valid, dem, 0.0), sigma, mode="nearest") / np.maximum(weight, 1e-12)
    gy, gx = np.gradient(np.where(weight > 0.2, smooth, 0.0), pixel_size, pixel_size)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    slope[weight <= 0.2] = np.nan
    return slope


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT / "config/nac_m173.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--downsample-factor", type=int, default=1)
    parser.add_argument("--normal-weight", type=float)
    parser.add_argument("--albedo-window", type=int)
    parser.add_argument("--lunar-lambert-l", type=float)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    completion = args.output / "run-METRICS.json"
    if completion.exists() and not args.force:
        print(f"REUSE {completion}")
        print(completion.read_text(encoding="utf-8"))
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    document = load_document(args.config)
    settings = dict(document["safs"])
    overrides = {
        "normal_weight": args.normal_weight,
        "albedo_window_final_px": args.albedo_window,
        "lunar_lambert_l": args.lunar_lambert_l,
        "iterations_per_level": args.iterations,
    }
    settings.update({key: value for key, value in overrides.items() if value is not None})
    config = SafsConfig(**settings)
    inputs = document["inputs"]

    with rasterio.open(inputs["image"]) as source:
        final_shape = (
            int(np.ceil(source.height / args.downsample_factor)),
            int(np.ceil(source.width / args.downsample_factor)),
        )
    image, profile = read_resampled(Path(inputs["image"]), 1, final_shape, Resampling.average)
    mask_values, _ = read_resampled(Path(inputs["valid_mask"]), 1, final_shape, Resampling.average)
    valid = np.isfinite(image) & np.isfinite(mask_values) & (mask_values >= 0.75)
    radiometric_valid = valid & (image > 0.0)
    input_scale = float(np.median(image[radiometric_valid]))
    normalized_image = image / input_scale
    with rasterio.open(inputs["coarse_dem"]) as source:
        coarse = source.read(1).astype(np.float64)
        if source.nodata is not None:
            coarse[np.isclose(coarse, source.nodata)] = np.nan
    bands = inputs["geometry_bands"]
    incidence, _ = read_resampled(Path(inputs["geometry"]), int(bands["incidence"]), final_shape, Resampling.bilinear)
    emission, _ = read_resampled(Path(inputs["geometry"]), int(bands["emission"]), final_shape, Resampling.bilinear)
    sun_azimuth, _ = read_resampled(Path(inputs["geometry"]), int(bands["sun_azimuth"]), final_shape, Resampling.bilinear)
    spacecraft_azimuth, _ = read_resampled(Path(inputs["geometry"]), int(bands["spacecraft_azimuth"]), final_shape, Resampling.bilinear)
    sun = direction_from_azimuth_zenith(sun_azimuth, incidence)
    view = direction_from_azimuth_zenith(spacecraft_azimuth, emission)
    pixel_x = abs(float(profile["transform"].a))
    pixel_y = abs(float(profile["transform"].e))

    def progress(row: dict[str, float]) -> None:
        print(
            f"level={int(row['level'])} shape={int(row['rows'])}x{int(row['columns'])} "
            f"iter={int(row['iteration'])} cost={row['cost_before']:.8g}->{row['cost_after']:.8g} "
            f"step={row['maximum_height_step_m']:.6g} accept={row['accepted_fraction']:.3f}",
            flush=True,
        )

    result = solve_safs(
        normalized_image, coarse, pixel_x, pixel_y, sun, view,
        config=config, valid_mask=valid, progress=progress,
    )
    initial = resize_bilinear(coarse, final_shape)
    modeled_iof = result.modeled_intensity * input_scale
    photo_valid = result.valid_mask & radiometric_valid & np.isfinite(modeled_iof)
    photo_residual = modeled_iof - image
    metrics = {
        "method": "Independent Wu et al. (2016) SAfS reproduction",
        "not_author_source_code": True,
        "downsample_factor": args.downsample_factor,
        "pixel_size_x_m": pixel_x,
        "pixel_size_y_m": pixel_y,
        "shape": list(final_shape),
        "input_image_median_iof": input_scale,
        "pyramid_shapes": [list(shape) for shape in result.pyramid_shapes],
        **error_stats(photo_residual, photo_valid, "self_photo"),
        "dem_update_rmse_m": float(np.sqrt(np.nanmean((result.dem[valid] - initial[valid]) ** 2))),
        "dem_update_p95_abs_m": float(np.nanpercentile(np.abs(result.dem[valid] - initial[valid]), 95.0)),
        "albedo_mean": float(np.nanmean(result.albedo[valid])),
        "albedo_std": float(np.nanstd(result.albedo[valid])),
    }

    reference = reproject_reference(REFERENCE, profile)
    reference_mask = reproject_reference(REFERENCE_MASK, profile, nearest=True)
    geometry_valid = valid & np.isfinite(result.dem) & np.isfinite(reference) & (reference_mask > 0.5)
    if geometry_valid.sum() >= 1000:
        dz = float(np.median(reference[geometry_valid] - result.dem[geometry_valid]))
        aligned = result.dem + dz
        elevation_residual = aligned - reference
        metrics["vertical_shift_to_stereo_m"] = dz
        metrics.update(error_stats(elevation_residual, geometry_valid, "stereo_elevation"))
        ref_slope = smooth_slope(reference, geometry_valid, pixel_x)
        model_slope = smooth_slope(aligned, geometry_valid, pixel_x)
        slope_valid = geometry_valid & np.isfinite(ref_slope) & np.isfinite(model_slope)
        metrics.update(error_stats(model_slope - ref_slope, slope_valid, "stereo_slope20m_deg"))
        write_raster(args.output / "run-ELEVATION-MINUS-STEREO.tif", np.where(geometry_valid, elevation_residual, np.nan), profile)
    else:
        metrics["stereo_elevation_pixels"] = int(geometry_valid.sum())

    write_raster(args.output / "run-DEM.tif", result.dem, profile)
    write_raster(args.output / "run-ALBEDO.tif", result.albedo, profile)
    write_raster(args.output / "run-MODELED-IOF.tif", modeled_iof, profile)
    write_raster(args.output / "run-PHOTOMETRIC-RESIDUAL.tif", np.where(photo_valid, photo_residual, np.nan), profile)
    write_raster(args.output / "run-DEM-MINUS-INITIAL.tif", result.dem - initial, profile)
    write_raster(args.output / "run-VALID-MASK.tif", valid.astype(np.uint8), profile, byte=True)
    with (args.output / "run-HISTORY.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result.history[0]))
        writer.writeheader()
        writer.writerows(result.history)
    metadata = {
        "config_file": str(args.config), "inputs": inputs,
        "config": result.config, "metrics": metrics,
    }
    (args.output / "run-METADATA.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    completion.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    update = result.dem - initial
    residual_limit = max(float(np.nanpercentile(np.abs(photo_residual[photo_valid]), 99)), 1e-9)
    update_limit = max(float(np.nanpercentile(np.abs(update[valid]), 99)), 1e-9)
    image_limits = np.percentile(image[valid], (1, 99))
    panels = [
        ("Observed NAC I/F", image, "gray", image_limits[0], image_limits[1]),
        ("Modeled I/F", modeled_iof, "gray", image_limits[0], image_limits[1]),
        ("Photometric residual", photo_residual, "RdBu_r", -residual_limit, residual_limit),
        ("DEM update (m)", update, "RdBu_r", -update_limit, update_limit),
        ("Estimated albedo", result.albedo, "gray", *np.nanpercentile(result.albedo[valid], (1, 99))),
        ("Refined DEM", result.dem, "terrain", *np.nanpercentile(result.dem[valid], (1, 99))),
    ]
    for axis, (title, values, cmap, vmin, vmax) in zip(axes.flat, panels):
        artist = axis.imshow(np.where(valid, values, np.nan), cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")
        figure.colorbar(artist, ax=axis, shrink=0.72)
    figure.savefig(args.output / "run-OVERVIEW.png", dpi=180)
    plt.close(figure)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
