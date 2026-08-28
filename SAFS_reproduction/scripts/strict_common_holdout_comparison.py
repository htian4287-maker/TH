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


ROOT = Path("/mnt/e/NAC_Photometry/paper2016_multi")
OUTPUT = ROOT / "27_safs_single_vs_multi/03_cross_illumination_holdout"
IMAGE_ROOT = ROOT / "13_grumpe_validation/03_geometry/full9/images_2m"
OTHER_RESIDUALS = ROOT / "23_asp_hapke_clean8_eightfold/05_comparison/residuals"


def read(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as source:
        values = source.read(1).astype(np.float64)
        if source.nodata is not None:
            values[np.isclose(values, source.nodata)] = np.nan
        return values, source.profile.copy()


def on_profile(path: Path, profile: dict) -> np.ndarray:
    destination = np.full((profile["height"], profile["width"]), np.nan, dtype=np.float64)
    with rasterio.open(path) as source:
        reproject(
            source=rasterio.band(source, 1), destination=destination,
            src_transform=source.transform, src_crs=source.crs, src_nodata=source.nodata,
            dst_transform=profile["transform"], dst_crs=profile["crs"], dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    return destination


def stats(observed: np.ndarray, residual: np.ndarray, valid: np.ndarray) -> dict[str, float | int]:
    values = residual[valid]
    predicted = observed[valid] + values
    return {
        "common_pixels": int(valid.sum()),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "mae": float(np.mean(np.abs(values))),
        "bias": float(np.mean(values)),
        "nrmse_median": float(np.sqrt(np.mean(values**2)) / np.median(observed[valid])),
        "correlation": float(np.corrcoef(observed[valid], predicted)[0, 1]),
        "p95_abs_residual": float(np.percentile(np.abs(values), 95)),
    }


def main() -> int:
    scenes = sorted(path.name.replace("_safs_residual.tif", "") for path in OUTPUT.glob("*_safs_residual.tif"))
    rows: list[dict] = []
    for scene in scenes:
        observed, profile = read(IMAGE_ROOT / f"{scene}.tif")
        residuals = {
            "SAfS single-M173": on_profile(OUTPUT / f"{scene}_safs_residual.tif", profile),
            "ASP Hapke multi-7": on_profile(OTHER_RESIDUALS / f"{scene}_asp_hapke_residual.tif", profile),
            "GRUMPE AMSA multi-7": on_profile(OTHER_RESIDUALS / f"{scene}_grumpe_amsa_residual.tif", profile),
        }
        common = np.isfinite(observed) & (observed > 0)
        for residual in residuals.values():
            common &= np.isfinite(residual)
        for model, residual in residuals.items():
            rows.append({
                "holdout_id": scene,
                "model": model,
                "training_images": 1 if model.startswith("SAfS") else 7,
                **stats(observed, residual, common),
            })

    with (OUTPUT / "strict_common_mask_holdout_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    models = sorted({row["model"] for row in rows})
    summary_rows = []
    for model in models:
        subset = [row for row in rows if row["model"] == model]
        summary_rows.append({
            "model": model,
            "folds": len(subset),
            "training_images": subset[0]["training_images"],
            "rmse_mean": float(np.mean([row["rmse"] for row in subset])),
            "rmse_median": float(np.median([row["rmse"] for row in subset])),
            "correlation_mean": float(np.mean([row["correlation"] for row in subset])),
            "nrmse_mean": float(np.mean([row["nrmse_median"] for row in subset])),
        })
    wins = {model: 0 for model in models}
    for scene in scenes:
        subset = [row for row in rows if row["holdout_id"] == scene]
        wins[min(subset, key=lambda row: row["rmse"])["model"]] += 1
    summary = {
        "protocol": "pixelwise intersection of SAfS, ASP and GRUMPE residual validity for each of seven non-M173 holdouts",
        "scenes": scenes,
        "model_summary": summary_rows,
        "rmse_wins": wins,
    }
    (OUTPUT / "strict_common_mask_holdout_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fig, axis = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    x = np.arange(len(scenes))
    for model in models:
        axis.plot(x, [next(row["rmse"] for row in rows if row["holdout_id"] == scene and row["model"] == model) for scene in scenes], marker="o", label=model)
    axis.set_xticks(x, scenes, rotation=35, ha="right", fontsize=8)
    axis.set(ylabel="I/F RMSE on strict common pixels", title="Single-image SAfS vs multi-image models: seven unseen illuminations")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(OUTPUT / "strict_common_mask_holdout_comparison.png", dpi=190)
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
