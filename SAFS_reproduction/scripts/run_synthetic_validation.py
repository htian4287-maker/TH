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
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from safs_method import SafsConfig, solve_safs  # noqa: E402
from safs_method.model import resize_bilinear  # noqa: E402
from safs_method.synthetic import make_synthetic_case  # noqa: E402


def load_config(path: Path) -> SafsConfig:
    values = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("safs", {})
    return SafsConfig(**values)


def rmse(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a[valid] - b[valid]) ** 2)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT / "config/synthetic.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/synthetic")
    parser.add_argument("--size", type=int, default=96)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    image, coarse, truth_dem, truth_albedo, sun, view = make_synthetic_case(args.size)
    config = load_config(args.config)
    result = solve_safs(image, coarse, 1.0, 1.0, sun, view, config)
    baseline = resize_bilinear(coarse, truth_dem.shape)
    # Geometry is assessed on the full known synthetic support, including
    # shadows.  result.valid_mask is intentionally the photometric mask only.
    valid = np.isfinite(result.dem) & np.isfinite(truth_dem)
    truth_albedo_normalized = truth_albedo / np.median(truth_albedo[valid])
    metrics = {
        "size": args.size,
        "coarse_shape": list(coarse.shape),
        "pyramid_shapes": [list(shape) for shape in result.pyramid_shapes],
        "baseline_dem_rmse_m": rmse(baseline, truth_dem, valid),
        "safs_dem_rmse_m": rmse(result.dem, truth_dem, valid),
        "dem_rmse_improvement_percent": 100.0 * (
            1.0 - rmse(result.dem, truth_dem, valid) / rmse(baseline, truth_dem, valid)
        ),
        "albedo_rmse": rmse(result.albedo, truth_albedo_normalized, valid),
        "albedo_correlation": float(np.corrcoef(result.albedo[valid], truth_albedo_normalized[valid])[0, 1]),
        "modeled_image_rmse": rmse(result.modeled_intensity, image, result.valid_mask),
        "history_records": len(result.history),
    }
    np.savez_compressed(
        args.output / "synthetic_products.npz",
        image=image,
        coarse_dem=coarse,
        baseline_dem=baseline,
        truth_dem=truth_dem,
        truth_albedo=truth_albedo_normalized,
        reconstructed_dem=result.dem,
        estimated_albedo=result.albedo,
        modeled_intensity=result.modeled_intensity,
        valid_mask=result.valid_mask,
    )
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (args.output / "history.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result.history[0]))
        writer.writeheader()
        writer.writerows(result.history)

    figure, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    panels = [
        ("Synthetic image", image, "gray"),
        ("Truth DEM", truth_dem, "terrain"),
        ("Interpolated coarse DEM", baseline, "terrain"),
        ("SAfS DEM", result.dem, "terrain"),
        ("Truth albedo", truth_albedo_normalized, "gray"),
        ("Estimated albedo", result.albedo, "gray"),
        ("Baseline error", baseline - truth_dem, "RdBu_r"),
        ("SAfS error", result.dem - truth_dem, "RdBu_r"),
    ]
    for axis, (title, values, cmap) in zip(axes.flat, panels):
        if "error" in title.lower():
            limit = max(float(np.nanpercentile(np.abs(values), 99)), 1e-9)
            image_artist = axis.imshow(values, cmap=cmap, vmin=-limit, vmax=limit)
        else:
            image_artist = axis.imshow(values, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
        figure.colorbar(image_artist, ax=axis, shrink=0.72)
    figure.savefig(args.output / "synthetic_validation.png", dpi=180)
    plt.close(figure)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
