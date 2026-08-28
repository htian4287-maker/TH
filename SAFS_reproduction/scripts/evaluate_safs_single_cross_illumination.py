#!/usr/bin/env python3
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


CODE = Path("/mnt/e/光度法代码/SAFS方法")
sys.path.insert(0, str(CODE / "src"))
from safs_method.model import (  # noqa: E402
    direction_from_azimuth_zenith,
    render_reflectance,
)


ROOT = Path("/mnt/e/NAC_Photometry/paper2016_multi")
SAFS_DEM = ROOT / "26_safs_reproduction/02_m173_full/recommended_nw1_aw61_ll0p35/run-DEM.tif"
SAFS_ALBEDO = ROOT / "26_safs_reproduction/02_m173_full/recommended_nw1_aw61_ll0p35/run-ALBEDO.tif"
IMAGE_ROOT = ROOT / "13_grumpe_validation/03_geometry/full9/images_2m"
GEOMETRY_ROOT = ROOT / "13_grumpe_validation/03_geometry/ba_pixel_geometry_full9_direct2m"
EXISTING_METRICS = ROOT / "23_asp_hapke_clean8_eightfold/05_comparison/eightfold_metrics_long.csv"
OUTPUT = ROOT / "27_safs_single_vs_multi/03_cross_illumination_holdout"
TRAIN_IMAGE = "M173246166LE"


def read(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as source:
        values = source.read(1).astype(np.float64)
        if source.nodata is not None:
            values[np.isclose(values, source.nodata)] = np.nan
        return values, source.profile.copy()


def reproject_to(path: Path, profile: dict) -> np.ndarray:
    destination = np.full((profile["height"], profile["width"]), np.nan, dtype=np.float64)
    with rasterio.open(path) as source:
        reproject(
            source=rasterio.band(source, 1), destination=destination,
            src_transform=source.transform, src_crs=source.crs, src_nodata=source.nodata,
            dst_transform=profile["transform"], dst_crs=profile["crs"], dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    return destination


def write(path: Path, values: np.ndarray, profile: dict) -> None:
    output = profile.copy()
    output.pop("blockxsize", None)
    output.pop("blockysize", None)
    output.update(driver="GTiff", count=1, dtype="float32", nodata=-9999.0, compress="deflate", tiled=False)
    with rasterio.open(path, "w", **output) as target:
        target.write(np.where(np.isfinite(values), values, -9999.0).astype("float32"), 1)


def metrics(observed: np.ndarray, predicted: np.ndarray, valid: np.ndarray) -> dict[str, float | int]:
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


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    existing = list(csv.DictReader(EXISTING_METRICS.open(encoding="utf-8-sig")))
    holdouts = sorted({row["holdout_id"] for row in existing if row["holdout_id"] != TRAIN_IMAGE})
    safs_rows: list[dict] = []
    residual_images: list[tuple[str, np.ndarray, np.ndarray]] = []
    for scene in holdouts:
        image_path = IMAGE_ROOT / f"{scene}.tif"
        geometry_dir = GEOMETRY_ROOT / scene / "maps_2m"
        required = {
            "incidence": geometry_dir / f"{scene}_local_incidence_2m.tif",
            "emission": geometry_dir / f"{scene}_local_emission_2m.tif",
            "sun_azimuth": geometry_dir / f"{scene}_sun_azimuth_2m.tif",
            "spacecraft_azimuth": geometry_dir / f"{scene}_spacecraft_azimuth_2m.tif",
        }
        if not image_path.exists() or not all(path.exists() for path in required.values()):
            print(f"SKIP missing inputs: {scene}", flush=True)
            continue
        observed, profile = read(image_path)
        dem = reproject_to(SAFS_DEM, profile)
        albedo = reproject_to(SAFS_ALBEDO, profile)
        geometry = {name: read(path)[0] for name, path in required.items()}
        sun = direction_from_azimuth_zenith(geometry["sun_azimuth"], geometry["incidence"])
        view = direction_from_azimuth_zenith(geometry["spacecraft_azimuth"], geometry["emission"])
        reflectance, illuminated = render_reflectance(dem, 2.0, 2.0, sun, view, 0.35)
        base = albedo * reflectance
        valid = (
            np.isfinite(observed) & (observed > 0.0) & np.isfinite(base) & (base > 0.0)
            & np.isfinite(dem) & illuminated
        )
        if valid.sum() < 1000:
            print(f"SKIP too few valid pixels: {scene}", flush=True)
            continue
        shadow_cutoff = float(np.percentile(observed[valid], 1.0))
        valid &= observed > shadow_cutoff
        gain = float(np.sum(observed[valid] * base[valid]) / np.sum(base[valid] ** 2))
        prediction = gain * base
        row = {
            "holdout_id": scene,
            "model": "SAfS single-M173",
            "training_images": 1,
            "training_image_ids": TRAIN_IMAGE,
            "gain": gain,
            **metrics(observed, prediction, valid),
        }
        safs_rows.append(row)
        residual = np.where(valid, prediction - observed, np.nan)
        write(OUTPUT / f"{scene}_safs_prediction.tif", np.where(valid, prediction, np.nan), profile)
        write(OUTPUT / f"{scene}_safs_residual.tif", residual, profile)
        residual_images.append((scene, observed, residual))
        print(json.dumps(row, ensure_ascii=False), flush=True)

    long_rows: list[dict] = []
    safs_by_scene = {row["holdout_id"]: row for row in safs_rows}
    for scene in sorted(safs_by_scene):
        long_rows.append(safs_by_scene[scene])
        for row in existing:
            if row["holdout_id"] == scene and row["model"] in {"ASP Hapke", "GRUMPE AMSA"}:
                long_rows.append({
                    "holdout_id": scene,
                    "model": row["model"] + " multi-7",
                    "training_images": int(row["training_images"]),
                    "training_image_ids": "seven scenes excluding holdout",
                    "gain": float(row["gain"]),
                    "valid_pixels": int(row["valid_pixels"]),
                    "rmse": float(row["rmse"]),
                    "mae": float(row["mae"]),
                    "bias": float(row["bias"]),
                    "nrmse_median": float(row["nrmse_median"]),
                    "correlation": float(row["correlation"]),
                    "p95_abs_residual": float(row["p95_abs_residual"]),
                })
    fields = list(long_rows[0])
    with (OUTPUT / "single_vs_multi_holdout_metrics_long.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(long_rows)

    models = sorted({row["model"] for row in long_rows})
    summaries = []
    for model in models:
        subset = [row for row in long_rows if row["model"] == model]
        summaries.append({
            "model": model,
            "folds": len(subset),
            "training_images": subset[0]["training_images"],
            "rmse_mean": float(np.mean([float(row["rmse"]) for row in subset])),
            "rmse_median": float(np.median([float(row["rmse"]) for row in subset])),
            "correlation_mean": float(np.mean([float(row["correlation"]) for row in subset])),
            "nrmse_mean": float(np.mean([float(row["nrmse_median"]) for row in subset])),
        })
    wins = {model: 0 for model in models}
    for scene in sorted(safs_by_scene):
        scene_rows = [row for row in long_rows if row["holdout_id"] == scene]
        wins[min(scene_rows, key=lambda row: float(row["rmse"]))["model"]] += 1
    summary = {
        "protocol": "SAfS trained only on M173; evaluated on seven other scenes. ASP/GRUMPE use their existing seven-image leave-one-out folds.",
        "scenes": sorted(safs_by_scene),
        "model_summary": summaries,
        "rmse_wins": wins,
    }
    (OUTPUT / "single_vs_multi_holdout_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    x = np.arange(len(safs_by_scene))
    scenes = sorted(safs_by_scene)
    for model in models:
        values = [next(float(row["rmse"]) for row in long_rows if row["holdout_id"] == scene and row["model"] == model) for scene in scenes]
        axes[0].plot(x, values, marker="o", label=model)
    axes[0].set_xticks(x, scenes, rotation=35, ha="right", fontsize=8)
    axes[0].set(ylabel="Holdout I/F RMSE", title="Cross-illumination holdout prediction")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].bar([row["model"] for row in summaries], [row["rmse_mean"] for row in summaries])
    axes[1].set(ylabel="Mean holdout I/F RMSE", title="One-image SAfS vs seven-image models")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(OUTPUT / "single_vs_multi_holdout_comparison.png", dpi=190)
    plt.close(fig)

    if residual_images:
        fig, axes = plt.subplots(len(residual_images), 2, figsize=(8, 3.2 * len(residual_images)), constrained_layout=True)
        axes = np.atleast_2d(axes)
        for row_index, (scene, observed, residual) in enumerate(residual_images):
            valid = np.isfinite(residual)
            lim = max(float(np.nanpercentile(np.abs(residual[valid]), 99)), 1e-9)
            axes[row_index, 0].imshow(observed, cmap="gray", vmin=np.nanpercentile(observed, 1), vmax=np.nanpercentile(observed, 99))
            axes[row_index, 1].imshow(residual, cmap="RdBu_r", vmin=-lim, vmax=lim)
            axes[row_index, 0].set_title(f"{scene} observed")
            axes[row_index, 1].set_title(f"SAfS(M173) residual, ±{lim:.4g}")
            axes[row_index, 0].axis("off")
            axes[row_index, 1].axis("off")
        fig.savefig(OUTPUT / "safs_cross_illumination_residual_atlas.png", dpi=160)
        plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
