#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from safs_method import SafsConfig, solve_safs  # noqa: E402
from safs_method.model import resize_bilinear  # noqa: E402
from safs_method.synthetic import make_synthetic_case  # noqa: E402


def main() -> int:
    output = PROJECT / "outputs/synthetic_tuning"
    output.mkdir(parents=True, exist_ok=True)
    image, coarse, truth, truth_albedo, sun, view = make_synthetic_case(96)
    baseline = resize_bilinear(coarse, truth.shape)
    valid = np.isfinite(truth)
    baseline_rmse = float(np.sqrt(np.mean((baseline[valid] - truth[valid]) ** 2)))
    rows = []
    for normal_weight in (0.0, 0.02, 0.1, 1.0, 10.0):
        for albedo_window in (31, 61, 95):
            for maximum_step in (0.05, 0.1):
                config = SafsConfig(
                    iterations_per_level=4,
                    maximum_pyramid_levels=9,
                    normal_weight=normal_weight,
                    albedo_window_final_px=albedo_window,
                    maximum_height_step_m=maximum_step,
                )
                result = solve_safs(image, coarse, 1.0, 1.0, sun, view, config)
                active = np.isfinite(result.dem)
                dem_rmse = float(np.sqrt(np.mean((result.dem[active] - truth[active]) ** 2)))
                photo = result.valid_mask & np.isfinite(result.modeled_intensity)
                truth_a = truth_albedo / np.median(truth_albedo[photo])
                rows.append({
                    "normal_weight": normal_weight,
                    "albedo_window": albedo_window,
                    "maximum_height_step_m": maximum_step,
                    "baseline_rmse_m": baseline_rmse,
                    "dem_rmse_m": dem_rmse,
                    "improvement_percent": 100.0 * (1.0 - dem_rmse / baseline_rmse),
                    "albedo_rmse": float(np.sqrt(np.mean((result.albedo[photo] - truth_a[photo]) ** 2))),
                    "albedo_correlation": float(np.corrcoef(result.albedo[photo], truth_a[photo])[0, 1]),
                    "photo_rmse": float(np.sqrt(np.mean((result.modeled_intensity[photo] - image[photo]) ** 2))),
                })
                print(rows[-1], flush=True)
    rows.sort(key=lambda row: row["dem_rmse_m"])
    with (output / "grid.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "best.json").write_text(json.dumps(rows[0], indent=2), encoding="utf-8")
    print("BEST", json.dumps(rows[0], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
