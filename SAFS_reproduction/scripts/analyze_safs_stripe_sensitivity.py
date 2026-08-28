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


ROOT = Path("/mnt/e/NAC_Photometry/paper2016_multi/26_safs_reproduction/01_m173_2m_sensitivity")
OUTPUT = Path("/mnt/e/NAC_Photometry/paper2016_multi/27_safs_single_vs_multi/01_safs_stripe_ablation")


def read(path: Path) -> np.ndarray:
    with rasterio.open(path) as source:
        values = source.read(1).astype(np.float64)
        if source.nodata is not None:
            values[np.isclose(values, source.nodata)] = np.nan
    return values


def axis_means(values: np.ndarray, valid: np.ndarray, axis: int) -> np.ndarray:
    sums = np.sum(np.where(valid, values, 0.0), axis=axis)
    counts = np.sum(valid, axis=axis)
    return np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)


def stripe_metrics(update: np.ndarray, mask: np.ndarray, pixel_m: float = 2.0) -> dict[str, float]:
    valid = binary_erosion(mask & np.isfinite(update), iterations=5)
    weights = gaussian_filter(valid.astype(np.float64), 10.0 / pixel_m, mode="nearest")
    low = gaussian_filter(np.where(valid, update, 0.0), 10.0 / pixel_m, mode="nearest")
    low = np.divide(low, weights, out=np.full_like(low, np.nan), where=weights > 0.2)
    high = update - low
    high_valid = valid & np.isfinite(high)
    row_profile = axis_means(high, high_valid, axis=1)
    column_profile = axis_means(high, high_valid, axis=0)
    row_proxy = float(np.nanstd(row_profile))
    column_proxy = float(np.nanstd(column_profile))
    return {
        "stripe_valid_pixels": int(high_valid.sum()),
        "update_rmse_m": float(np.sqrt(np.mean(update[valid] ** 2))),
        "update_highpass_rmse_m": float(np.sqrt(np.mean(high[high_valid] ** 2))),
        "row_stripe_proxy_m": row_proxy,
        "column_stripe_proxy_m": column_proxy,
        "column_to_row_anisotropy": column_proxy / max(row_proxy, 1e-12),
        "column_profile_p90_range_m": float(np.nanpercentile(column_profile, 95) - np.nanpercentile(column_profile, 5)),
    }


def pareto(rows: list[dict], keys: tuple[str, ...]) -> set[str]:
    result: set[str] = set()
    for candidate in rows:
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            if all(float(other[key]) <= float(candidate[key]) for key in keys) and any(
                float(other[key]) < float(candidate[key]) for key in keys
            ):
                dominated = True
                break
        if not dominated:
            result.add(str(candidate["case"]))
    return result


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for case_dir in sorted((ROOT / "grid").iterdir()):
        metrics_path = case_dir / "run-METRICS.json"
        update_path = case_dir / "run-DEM-MINUS-INITIAL.tif"
        mask_path = case_dir / "run-VALID-MASK.tif"
        if not (metrics_path.exists() and update_path.exists() and mask_path.exists()):
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        update = read(update_path)
        mask = read(mask_path) > 0.5
        metadata = json.loads((case_dir / "run-METADATA.json").read_text(encoding="utf-8"))
        config = metadata["config"]
        row = {
            "case": case_dir.name,
            "normal_weight": config["normal_weight"],
            "albedo_window": config["albedo_window_final_px"],
            "lunar_lambert_l": config["lunar_lambert_l"],
            "stereo_elevation_rmse_m": metrics["stereo_elevation_rmse"],
            "stereo_slope20m_rmse_deg": metrics["stereo_slope20m_deg_rmse"],
            "self_photo_rmse": metrics["self_photo_rmse"],
            **stripe_metrics(update, mask),
        }
        rows.append(row)

    objectives = (
        "stereo_elevation_rmse_m",
        "stereo_slope20m_rmse_deg",
        "self_photo_rmse",
        "column_stripe_proxy_m",
    )
    frontier = pareto(rows, objectives)
    minima = {key: min(float(row[key]) for row in rows) for key in objectives}
    for row in rows:
        row["pareto_geometry_photo_stripe"] = row["case"] in frontier
        row["balanced_four_objective_score"] = float(np.mean([
            float(row[key]) / minima[key] for key in objectives
        ]))
    recommended = min(
        (row for row in rows if row["case"] in frontier),
        key=lambda row: float(row["balanced_four_objective_score"]),
    )
    rows.sort(key=lambda row: float(row["balanced_four_objective_score"]))
    with (OUTPUT / "safs_2m_geometry_stripe_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "cases_analyzed": len(rows),
        "selection_rule": "minimum equal-weight normalized geometry+slope+self-photometry+column-stripe score on Pareto frontier",
        "pareto_cases": sorted(frontier),
        "stripe_aware_recommended": recommended,
        "previous_geometry_recommended": "nw1p0_aw61_ll0p35",
    }
    (OUTPUT / "safs_stripe_ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    colors = [float(row["self_photo_rmse"]) for row in rows]
    scatter = axes[0].scatter(
        [float(row["stereo_elevation_rmse_m"]) for row in rows],
        [float(row["column_stripe_proxy_m"]) for row in rows],
        c=colors, cmap="viridis", s=90, edgecolor="black",
    )
    for row in rows:
        if row["case"] in frontier:
            axes[0].annotate(str(row["case"]), (row["stereo_elevation_rmse_m"], row["column_stripe_proxy_m"]), fontsize=7)
    fig.colorbar(scatter, ax=axes[0], label="Self-photo RMSE")
    axes[0].set(xlabel="Elevation RMSE vs stereo DTM (m)", ylabel="Column stripe proxy (m)", title="Geometry–stripe trade-off")
    axes[1].scatter(
        [float(row["stereo_slope20m_rmse_deg"]) for row in rows],
        [float(row["column_to_row_anisotropy"]) for row in rows],
        c=[float(row["normal_weight"]) for row in rows], cmap="plasma", s=90, edgecolor="black",
    )
    for row in rows:
        axes[1].annotate(str(row["case"]), (row["stereo_slope20m_rmse_deg"], row["column_to_row_anisotropy"]), fontsize=6)
    axes[1].set(xlabel="20 m slope RMSE (deg)", ylabel="Column/row anisotropy", title="Slope–anisotropy trade-off")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.savefig(OUTPUT / "safs_2m_geometry_stripe_tradeoff.png", dpi=190)
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

