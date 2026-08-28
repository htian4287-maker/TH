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
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import binary_erosion, gaussian_filter


ROOT = Path("/mnt/e/NAC_Photometry/paper2016_multi")
REFERENCE = ROOT / "24_independent_stereo_validation/02_aligned/reference_rimasharp3_on_aligned_model_grid.tif"
STRICT_MASK = ROOT / "24_independent_stereo_validation/02_aligned/strict_common_mask.tif"
CONFIDENCE = ROOT / "24_independent_stereo_validation/02_aligned/reference_confidence_on_aligned_model_grid.tif"
MODEL_GRID = ROOT / "07_sfs_multi/03_pilot_dem/sldem_pilot_2m.tif"
OUTPUT = ROOT / "27_safs_single_vs_multi/02_geometry_common_benchmark"
CANDIDATES = [
    {
        "model": "Initial SLDEM",
        "path": ROOT / "07_sfs_multi/03_pilot_dem/sldem_pilot_2m.tif",
        "already_aligned": False,
        "class": "initial",
        "training_images": 0,
        "uses_M173": False,
    },
    {
        "model": "ASP multi-7 baseline",
        "path": ROOT / "23_asp_hapke_clean8_eightfold/folds/M173246166LE/02_sfs/run-DEM-final.tif",
        "already_aligned": False,
        "class": "multi-image held-out M173",
        "training_images": 7,
        "uses_M173": False,
    },
    {
        "model": "GRUMPE multi-7 AMSA",
        "path": ROOT / "20_grumpe_clean8_eightfold/folds/M173246166LE/amsa/reconstructed_dem.tif",
        "already_aligned": False,
        "class": "multi-image held-out M173",
        "training_images": 7,
        "uses_M173": False,
    },
    {
        "model": "ASP multi-7 strong initial",
        "path": ROOT / "25_asp_regularization_m173_holdout/configs/sw0p04_iw0p001/01_sfs/run-DEM-final.tif",
        "already_aligned": False,
        "class": "multi-image held-out M173",
        "training_images": 7,
        "uses_M173": False,
    },
    {
        "model": "SAfS single M173",
        "path": ROOT / "26_safs_reproduction/02_m173_full/recommended_nw1_aw61_ll0p35/run-DEM.tif",
        "already_aligned": False,
        "class": "single-image fitted M173",
        "training_images": 1,
        "uses_M173": True,
    },
]


def read(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as source:
        values = source.read(1).astype(np.float64)
        if source.nodata is not None:
            values[np.isclose(values, source.nodata)] = np.nan
        return values, source.profile.copy()


def on_reference(candidate: dict, reference_profile: dict, model_profile: dict) -> np.ndarray:
    # The original independent validation first kept every model on the
    # unshifted 2 m model grid, then assigned the one shared dx/dy-corrected
    # geotransform.  Resampling a 0.5 m raster directly onto the shifted grid
    # changes pixel-centre phase by 0.75 m and biases steep-slope statistics.
    if candidate["already_aligned"]:
        values, _ = read(candidate["path"])
        if values.shape != (reference_profile["height"], reference_profile["width"]):
            raise ValueError(f"Aligned grid shape mismatch: {candidate['path']}")
        return values
    destination = np.full((model_profile["height"], model_profile["width"]), np.nan, dtype=np.float64)
    with rasterio.open(candidate["path"]) as source:
        reproject(
            source=rasterio.band(source, 1), destination=destination,
            src_transform=source.transform, src_crs=source.crs, src_nodata=source.nodata,
            dst_transform=model_profile["transform"], dst_crs=model_profile["crs"], dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    return destination


def error_stats(values: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    data = values[valid]
    median = float(np.median(data))
    return {
        "rmse": float(np.sqrt(np.mean(data**2))),
        "mae": float(np.mean(np.abs(data))),
        "bias": float(np.mean(data)),
        "nmad": float(1.4826 * np.median(np.abs(data - median))),
        "p95_abs": float(np.percentile(np.abs(data), 95)),
    }


def smooth(values: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    weight = gaussian_filter(valid.astype(np.float64), sigma, mode="nearest")
    result = gaussian_filter(np.where(valid, values, 0.0), sigma, mode="nearest")
    return np.divide(result, weight, out=np.full_like(result, np.nan), where=weight > 0.2)


def slope(values: np.ndarray, valid: np.ndarray, scale_m: float) -> np.ndarray:
    terrain = smooth(values, valid, max(scale_m / 4.0, 0.5))
    gy, gx = np.gradient(np.where(np.isfinite(terrain), terrain, 0.0), 2.0, 2.0)
    result = np.degrees(np.arctan(np.hypot(gx, gy)))
    result[~np.isfinite(terrain)] = np.nan
    return result


def axis_mean(values: np.ndarray, valid: np.ndarray, axis: int) -> np.ndarray:
    sums = np.sum(np.where(valid, values, 0.0), axis=axis)
    counts = np.sum(valid, axis=axis)
    return np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)


def stripe(update: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    low = smooth(update, valid, 5.0)
    high = update - low
    active = valid & np.isfinite(high)
    row = axis_mean(high, active, 1)
    column = axis_mean(high, active, 0)
    row_proxy = float(np.nanstd(row))
    column_proxy = float(np.nanstd(column))
    return {
        "update_rmse_m": float(np.sqrt(np.mean(update[valid] ** 2))),
        "update_highpass_rmse_m": float(np.sqrt(np.mean(high[active] ** 2))),
        "row_stripe_proxy_m": row_proxy,
        "column_stripe_proxy_m": column_proxy,
        "column_to_row_anisotropy": column_proxy / max(row_proxy, 1e-12),
    }


def write(path: Path, values: np.ndarray, profile: dict) -> None:
    output = profile.copy()
    output.pop("blockxsize", None)
    output.pop("blockysize", None)
    output.update(driver="GTiff", count=1, dtype="float32", nodata=-9999.0, compress="deflate", tiled=False)
    with rasterio.open(path, "w", **output) as target:
        target.write(np.where(np.isfinite(values), values, -9999.0).astype("float32"), 1)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reference, profile = read(REFERENCE)
    mask, _ = read(STRICT_MASK)
    confidence, _ = read(CONFIDENCE)
    _, model_profile = read(MODEL_GRID)
    source_values = {item["model"]: on_reference(item, profile, model_profile) for item in CANDIDATES}
    common = (mask > 0.5) & np.isfinite(reference)
    for values in source_values.values():
        common &= np.isfinite(values)

    # Match the earlier independent-validation protocol: fit dz on the
    # confidence 10–15 support eroded by three pixels, then evaluate on the
    # stricter common mask.  Existing rigid-aligned products therefore receive
    # an additional dz of approximately zero rather than being re-centred on a
    # different evaluation mask.
    fit_mask = np.isfinite(reference) & np.isfinite(confidence) & (confidence >= 10) & (confidence <= 15)
    for values in source_values.values():
        fit_mask &= np.isfinite(values)
    fit_mask = binary_erosion(fit_mask, iterations=3)

    # Refit one constant vertical offset for every model on exactly the same pixels.
    aligned: dict[str, np.ndarray] = {}
    for item in CANDIDATES:
        values = source_values[item["model"]]
        dz = float(np.median(reference[fit_mask] - values[fit_mask]))
        aligned[item["model"]] = values + dz
        item["dz_m"] = dz
    initial = aligned["Initial SLDEM"]
    ref_slopes = {scale: slope(reference, common, scale) for scale in (2.0, 10.0, 20.0, 40.0)}
    rows: list[dict] = []
    residuals: dict[str, np.ndarray] = {}
    updates: dict[str, np.ndarray] = {}
    for item in CANDIDATES:
        name = item["model"]
        residual = np.where(common, aligned[name] - reference, np.nan)
        update = np.where(common, aligned[name] - initial, np.nan)
        update -= np.nanmedian(update[common])
        residuals[name] = residual
        updates[name] = update
        row = {
            "model": name,
            "class": item["class"],
            "training_images": item["training_images"],
            "uses_M173_in_optimization": item["uses_M173"],
            "pixels": int(common.sum()),
            "vertical_shift_m": item["dz_m"],
        }
        row.update({f"elevation_{key}_m": value for key, value in error_stats(residual, common).items()})
        for scale, ref_slope in ref_slopes.items():
            model_slope = slope(aligned[name], common, scale)
            active = common & np.isfinite(model_slope) & np.isfinite(ref_slope)
            stat = error_stats(model_slope - ref_slope, active)
            row[f"slope_{int(scale)}m_rmse_deg"] = stat["rmse"]
            row[f"slope_{int(scale)}m_nmad_deg"] = stat["nmad"]
        row.update(stripe(update, common))
        rows.append(row)
        token = name.replace(" ", "_").replace("-", "m")
        write(OUTPUT / f"{token}_aligned.tif", aligned[name], profile)
        write(OUTPUT / f"{token}_minus_stereo.tif", residual, profile)
        write(OUTPUT / f"{token}_minus_initial.tif", update, profile)

    with (OUTPUT / "single_vs_multi_geometry_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": "same official stereo DTM, same shared dx/dy, per-model constant dz only, strict common mask",
        "resampling_protocol": "0.5 m SAfS first resampled to the original unshifted 2 m model grid; shared dx/dy then assigned identically to all models",
        "common_pixels": int(common.sum()),
        "vertical_fit_pixels": int(fit_mask.sum()),
        "information_asymmetry": "SAfS was fitted to M173, while the multi-image models held M173 out.",
        "geometry_winner": min(rows, key=lambda row: float(row["elevation_rmse_m"]))["model"],
        "slope20m_winner": min(rows, key=lambda row: float(row["slope_20m_rmse_deg"]))["model"],
        "models": rows,
    }
    (OUTPUT / "single_vs_multi_geometry_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    names = [item["model"] for item in CANDIDATES]
    fig, axes = plt.subplots(3, len(names), figsize=(4 * len(names), 11), constrained_layout=True)
    dem_min, dem_max = np.nanpercentile(reference[common], (2, 98))
    res_lim = max(float(np.nanpercentile(np.abs(residuals[name][common]), 98)) for name in names)
    update_lim = max(float(np.nanpercentile(np.abs(updates[name][common]), 98)) for name in names[1:])
    for column, name in enumerate(names):
        axes[0, column].imshow(aligned[name], cmap="terrain", vmin=dem_min, vmax=dem_max)
        axes[1, column].imshow(residuals[name], cmap="RdBu_r", vmin=-res_lim, vmax=res_lim)
        axes[2, column].imshow(updates[name], cmap="RdBu_r", vmin=-update_lim, vmax=update_lim)
        axes[0, column].set_title(name, fontsize=10)
        for row_index in range(3):
            axes[row_index, column].axis("off")
    fig.savefig(OUTPUT / "single_vs_multi_geometry_overview.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
