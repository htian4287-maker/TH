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
from scipy.ndimage import binary_erosion, gaussian_filter


ROOT = Path("/mnt/e/NAC_Photometry/paper2016_multi")
OUTPUT = ROOT / "27_safs_single_vs_multi/01_safs_stripe_ablation/03_full_comparison"
CASES = {
    "geometry_selected_nw1_aw61_L0.35": ROOT / "26_safs_reproduction/02_m173_full/recommended_nw1_aw61_ll0p35",
    "low_anisotropy_nw0.02_aw61_L0.55": ROOT / "27_safs_single_vs_multi/01_safs_stripe_ablation/02_full_nw0p02_aw61_ll0p55",
}


def read(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as source:
        values = source.read(1).astype(np.float64)
        if source.nodata is not None:
            values[np.isclose(values, source.nodata)] = np.nan
        return values, source.profile.copy()


def axis_mean(values: np.ndarray, valid: np.ndarray, axis: int) -> np.ndarray:
    sums = np.sum(np.where(valid, values, 0.0), axis=axis)
    counts = np.sum(valid, axis=axis)
    return np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = {}
    common = None
    for name, directory in CASES.items():
        update, _ = read(directory / "run-DEM-MINUS-INITIAL.tif")
        mask, _ = read(directory / "run-VALID-MASK.tif")
        valid = (mask > 0.5) & np.isfinite(update)
        data[name] = {"update": update, "valid": valid, "metrics": json.loads((directory / "run-METRICS.json").read_text(encoding="utf-8"))}
        common = valid.copy() if common is None else common & valid
    common = binary_erosion(common, iterations=20)
    rows = []
    profiles = {}
    highpasses = {}
    for name, item in data.items():
        update = item["update"]
        weight = gaussian_filter(common.astype(np.float64), 20.0, mode="nearest")
        low = gaussian_filter(np.where(common, update, 0.0), 20.0, mode="nearest")
        low = np.divide(low, weight, out=np.full_like(low, np.nan), where=weight > 0.2)
        high = update - low
        valid = common & np.isfinite(high)
        row_profile = axis_mean(high, valid, 1)
        column_profile = axis_mean(high, valid, 0)
        profiles[name] = column_profile
        highpasses[name] = np.where(valid, high, np.nan)
        row_proxy = float(np.nanstd(row_profile))
        column_proxy = float(np.nanstd(column_profile))
        metrics = item["metrics"]
        rows.append({
            "case": name,
            "common_pixels": int(common.sum()),
            "stereo_elevation_rmse_m": metrics["stereo_elevation_rmse"],
            "stereo_slope20m_rmse_deg": metrics["stereo_slope20m_deg_rmse"],
            "self_photo_rmse": metrics["self_photo_rmse"],
            "dem_update_rmse_m": float(np.sqrt(np.mean(update[common] ** 2))),
            "update_highpass_rmse_m": float(np.sqrt(np.mean(high[valid] ** 2))),
            "row_stripe_proxy_m": row_proxy,
            "column_stripe_proxy_m": column_proxy,
            "column_to_row_anisotropy": column_proxy / max(row_proxy, 1e-12),
            "column_profile_p90_range_m": float(np.nanpercentile(column_profile, 95) - np.nanpercentile(column_profile, 5)),
        })
    with (OUTPUT / "full_resolution_ablation_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    old, new = rows
    summary = {
        "comparison": rows,
        "changes_low_anisotropy_vs_geometry_selected_percent": {
            key: 100.0 * (float(new[key]) / float(old[key]) - 1.0)
            for key in (
                "stereo_elevation_rmse_m", "stereo_slope20m_rmse_deg", "self_photo_rmse",
                "update_highpass_rmse_m", "column_stripe_proxy_m", "column_to_row_anisotropy",
            )
        },
    }
    (OUTPUT / "full_resolution_ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    names = list(CASES)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    update_lim = max(float(np.nanpercentile(np.abs(data[name]["update"][common]), 99)) for name in names)
    high_lim = max(float(np.nanpercentile(np.abs(highpasses[name][common]), 99)) for name in names)
    for column, name in enumerate(names):
        axes[0, column].imshow(np.where(common, data[name]["update"], np.nan), cmap="RdBu_r", vmin=-update_lim, vmax=update_lim)
        axes[0, column].set_title(name + "\nDEM - initial")
        axes[1, column].imshow(highpasses[name], cmap="RdBu_r", vmin=-high_lim, vmax=high_lim)
        axes[1, column].set_title(name + "\n20 m high-pass update")
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    fig.savefig(OUTPUT / "full_resolution_update_comparison.png", dpi=190)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for name in names:
        axis.plot(np.arange(len(profiles[name])) * 0.5, profiles[name], label=name)
    axis.set(xlabel="East-west distance (m)", ylabel="Column-mean high-pass update (m)", title="Vertical stripe profiles")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(OUTPUT / "full_resolution_column_stripe_profiles.png", dpi=190)
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
