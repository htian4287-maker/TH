#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import gaussian_filter


REFERENCE = Path("/mnt/e/NAC_Photometry/paper2016_multi/24_independent_stereo_validation/02_aligned/reference_rimasharp3_on_aligned_model_grid.tif")
STRICT_MASK = Path("/mnt/e/NAC_Photometry/paper2016_multi/24_independent_stereo_validation/02_aligned/strict_common_mask.tif")
MODELS = {
    "Initial SLDEM": Path("/mnt/e/NAC_Photometry/paper2016/05_safs_inputs/sldem_initial_0p5m.tif"),
    "ASP single M173": Path("/mnt/e/NAC_Photometry/paper2016/08_asp_sfs_valid/full_shadow/run-DEM-final.tif"),
    "GRUMPE single M173": Path("/mnt/e/NAC_Photometry/paper2016_multi/13_grumpe_validation/05_integrate/single_M173_hapke_ba/G2_lowpass_r08_dem.tif"),
    "SAfS single M173": Path("/mnt/e/NAC_Photometry/paper2016_multi/26_safs_reproduction/02_m173_full/recommended_nw1_aw61_ll0p35/run-DEM.tif"),
}
DX_M = -34.0
DY_M = -30.0


def read(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as source:
        values = source.read(1).astype(np.float64)
        if source.nodata is not None:
            values[np.isclose(values, source.nodata)] = np.nan
        return values, source.profile.copy()


def shifted_on_reference(path: Path, reference_profile: dict) -> np.ndarray:
    destination = np.full(
        (reference_profile["height"], reference_profile["width"]),
        np.nan,
        dtype=np.float64,
    )
    with rasterio.open(path) as source:
        shifted = Affine(
            source.transform.a,
            source.transform.b,
            source.transform.c + DX_M,
            source.transform.d,
            source.transform.e,
            source.transform.f + DY_M,
        )
        reproject(
            source=rasterio.band(source, 1),
            destination=destination,
            src_transform=shifted,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=reference_profile["transform"],
            dst_crs=reference_profile["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    return destination


def stats(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    median = float(np.median(finite))
    return {
        "rmse": float(np.sqrt(np.mean(finite**2))),
        "mae": float(np.mean(np.abs(finite))),
        "bias": float(np.mean(finite)),
        "nmad": float(1.4826 * np.median(np.abs(finite - median))),
        "p95_abs": float(np.percentile(np.abs(finite), 95.0)),
    }


def weighted_smooth(values: np.ndarray, valid: np.ndarray, sigma_px: float) -> np.ndarray:
    weight = gaussian_filter(valid.astype(np.float64), sigma_px, mode="nearest")
    smooth = gaussian_filter(np.where(valid, values, 0.0), sigma_px, mode="nearest")
    return np.where(weight > 0.2, smooth / np.maximum(weight, 1e-12), np.nan)


def slope(values: np.ndarray, valid: np.ndarray, pixel_m: float, scale_m: float) -> np.ndarray:
    smooth = weighted_smooth(values, valid, max(scale_m / (2.0 * pixel_m), 0.5))
    gy, gx = np.gradient(np.where(np.isfinite(smooth), smooth, 0.0), pixel_m, pixel_m)
    result = np.degrees(np.arctan(np.hypot(gx, gy)))
    result[~np.isfinite(smooth)] = np.nan
    return result


def stripe_metrics(update: np.ndarray, valid: np.ndarray, pixel_m: float) -> dict[str, float]:
    low = weighted_smooth(update, valid, 10.0 / pixel_m)
    high = np.where(valid, update - low, np.nan)
    row_mean = np.nanmean(high, axis=1)
    column_mean = np.nanmean(high, axis=0)
    row_proxy = float(np.nanstd(row_mean))
    column_proxy = float(np.nanstd(column_mean))
    return {
        "update_rmse_m": float(np.sqrt(np.nanmean(update[valid] ** 2))),
        "update_highpass_rmse_m": float(np.sqrt(np.nanmean(high[valid] ** 2))),
        "row_stripe_proxy_m": row_proxy,
        "column_stripe_proxy_m": column_proxy,
        "column_to_row_anisotropy": column_proxy / max(row_proxy, 1e-12),
    }


def write_tif(path: Path, values: np.ndarray, profile: dict) -> None:
    output = profile.copy()
    output.pop("blockxsize", None)
    output.pop("blockysize", None)
    output.update(driver="GTiff", count=1, dtype="float32", nodata=-9999.0, compress="deflate", tiled=False)
    with rasterio.open(path, "w", **output) as target:
        target.write(np.where(np.isfinite(values), values, -9999.0).astype("float32"), 1)


def main() -> int:
    output = Path("/mnt/e/NAC_Photometry/paper2016_multi/26_safs_reproduction/03_single_model_comparison")
    output.mkdir(parents=True, exist_ok=True)
    reference, profile = read(REFERENCE)
    mask_raw, _ = read(STRICT_MASK)
    candidates = {name: shifted_on_reference(path, profile) for name, path in MODELS.items()}
    common = (mask_raw > 0.5) & np.isfinite(reference)
    for values in candidates.values():
        common &= np.isfinite(values)
    initial = candidates["Initial SLDEM"]
    rows: list[dict[str, object]] = []
    aligned: dict[str, np.ndarray] = {}
    residuals: dict[str, np.ndarray] = {}
    updates: dict[str, np.ndarray] = {}
    for name, values in candidates.items():
        dz = float(np.median(reference[common] - values[common]))
        model = values + dz
        residual = np.where(common, model - reference, np.nan)
        update = np.where(common, values - initial, np.nan)
        aligned[name] = model
        residuals[name] = residual
        updates[name] = update
        row: dict[str, object] = {"model": name, "pixels": int(common.sum()), "dz_m": dz}
        row.update({f"elevation_{key}_m": value for key, value in stats(residual).items()})
        for scale_m in (2.0, 10.0, 20.0, 40.0):
            ref_slope = slope(reference, common, 2.0, scale_m)
            model_slope = slope(model, common, 2.0, scale_m)
            slope_valid = common & np.isfinite(ref_slope) & np.isfinite(model_slope)
            values_stats = stats(np.where(slope_valid, model_slope - ref_slope, np.nan))
            row[f"slope_{int(scale_m)}m_rmse_deg"] = values_stats["rmse"]
            row[f"slope_{int(scale_m)}m_nmad_deg"] = values_stats["nmad"]
        row.update(stripe_metrics(update, common, 2.0))
        rows.append(row)
        write_tif(output / f"{name.replace(' ', '_')}_aligned.tif", model, profile)
        write_tif(output / f"{name.replace(' ', '_')}_minus_stereo.tif", residual, profile)
        write_tif(output / f"{name.replace(' ', '_')}_minus_initial.tif", update, profile)

    with (output / "single_m173_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "comparison_scope": "same M173 single image, same initial SLDEM, same official stereo DTM and shared dx/dy",
        "shared_horizontal_shift_m": {"east": DX_M, "north": DY_M},
        "common_pixels": int(common.sum()),
        "models": rows,
        "geometry_winner": min(rows, key=lambda row: float(row["elevation_rmse_m"]))["model"],
        "slope20m_winner": min(rows, key=lambda row: float(row["slope_20m_rmse_deg"]))["model"],
        "lowest_column_stripe_proxy": min(rows[1:], key=lambda row: float(row["column_stripe_proxy_m"]))["model"],
    }
    (output / "single_m173_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    names = list(candidates)
    figure, axes = plt.subplots(3, len(names), figsize=(4.1 * len(names), 11), constrained_layout=True)
    residual_limit = max(np.nanpercentile(np.abs(values[common]), 98) for values in residuals.values())
    update_limit = max(np.nanpercentile(np.abs(values[common]), 98) for name, values in updates.items() if name != "Initial SLDEM")
    for column, name in enumerate(names):
        axes[0, column].imshow(aligned[name], cmap="terrain", vmin=np.nanpercentile(reference[common], 2), vmax=np.nanpercentile(reference[common], 98))
        axes[0, column].set_title(name)
        axes[1, column].imshow(residuals[name], cmap="RdBu_r", vmin=-residual_limit, vmax=residual_limit)
        axes[2, column].imshow(updates[name], cmap="RdBu_r", vmin=-update_limit, vmax=update_limit)
        for row_index in range(3):
            axes[row_index, column].axis("off")
    axes[0, 0].set_ylabel("Aligned DEM")
    axes[1, 0].set_ylabel("DEM - stereo")
    axes[2, 0].set_ylabel("DEM - initial")
    figure.savefig(output / "single_m173_dem_comparison.png", dpi=190)
    plt.close(figure)

    center_row = reference.shape[0] // 2
    center_column = reference.shape[1] // 2
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].plot(np.arange(reference.shape[1]) * 2.0, reference[center_row], "k", lw=2, label="Stereo reference")
    axes[1].plot(np.arange(reference.shape[0]) * 2.0, reference[:, center_column], "k", lw=2, label="Stereo reference")
    for name in names:
        axes[0].plot(np.arange(reference.shape[1]) * 2.0, aligned[name][center_row], lw=1, label=name)
        axes[1].plot(np.arange(reference.shape[0]) * 2.0, aligned[name][:, center_column], lw=1, label=name)
    axes[0].set(title="East-west center profile", xlabel="Distance (m)", ylabel="Elevation (m)")
    axes[1].set(title="North-south center profile", xlabel="Distance (m)", ylabel="Elevation (m)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.savefig(output / "single_m173_profiles.png", dpi=190)
    plt.close(figure)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
