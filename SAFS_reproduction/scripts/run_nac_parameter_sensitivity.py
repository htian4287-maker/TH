#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]


def token(value: float | int) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def run_case(
    script: Path,
    config: Path,
    output: Path,
    normal_weight: float,
    albedo_window: int,
    lunar_lambert_l: float,
    iterations: int,
) -> dict:
    metrics_path = output / "run-METRICS.json"
    if not metrics_path.exists():
        command = [
            sys.executable, str(script), "--config", str(config),
            "--output", str(output), "--downsample-factor", "4",
            "--normal-weight", str(normal_weight),
            "--albedo-window", str(albedo_window),
            "--lunar-lambert-l", str(lunar_lambert_l),
            "--iterations", str(iterations),
        ]
        print("RUN", output.name, flush=True)
        subprocess.run(command, cwd=PROJECT, check=True)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "case": output.name,
        "normal_weight": normal_weight,
        "albedo_window": albedo_window,
        "lunar_lambert_l": lunar_lambert_l,
        "iterations_per_level": iterations,
        **metrics,
    }


def pareto(rows: list[dict], keys: tuple[str, ...]) -> list[str]:
    result = []
    for candidate in rows:
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            no_worse = all(float(other[key]) <= float(candidate[key]) for key in keys)
            better = any(float(other[key]) < float(candidate[key]) for key in keys)
            if no_worse and better:
                dominated = True
                break
        if not dominated:
            result.append(str(candidate["case"]))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT / "config/nac_m173.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runner = PROJECT / "scripts/run_nac_m173.py"
    rows: list[dict] = []

    for normal_weight in (0.02, 0.1, 1.0):
        for albedo_window in (31, 61, 95):
            name = f"nw{token(normal_weight)}_aw{albedo_window}_ll0p55"
            rows.append(run_case(
                runner, args.config, args.output / "grid" / name,
                normal_weight, albedo_window, 0.55, args.iterations,
            ))

    geometry_keys = ("stereo_elevation_rmse", "stereo_slope20m_deg_rmse")
    base_best = min(
        rows,
        key=lambda row: sum(
            float(row[key]) / min(float(candidate[key]) for candidate in rows)
            for key in geometry_keys
        ),
    )
    for lunar_lambert_l in (0.35, 0.75):
        name = (
            f"nw{token(base_best['normal_weight'])}_aw{base_best['albedo_window']}_"
            f"ll{token(lunar_lambert_l)}"
        )
        rows.append(run_case(
            runner, args.config, args.output / "grid" / name,
            float(base_best["normal_weight"]), int(base_best["albedo_window"]),
            lunar_lambert_l, args.iterations,
        ))

    objectives = (
        "stereo_elevation_rmse",
        "stereo_slope20m_deg_rmse",
        "self_photo_rmse",
    )
    frontier = pareto(rows, objectives)
    for row in rows:
        row["pareto_front"] = row["case"] in frontier
        row["geometry_ratio_score"] = float(np.mean([
            float(row[key]) / min(float(candidate[key]) for candidate in rows)
            for key in geometry_keys
        ]))
        row["balanced_ratio_score"] = float(np.mean([
            float(row[key]) / min(float(candidate[key]) for candidate in rows)
            for key in objectives
        ]))
    recommended = min(
        (row for row in rows if row["pareto_front"]),
        key=lambda row: float(row["geometry_ratio_score"]),
    )

    fields = list(rows[0].keys())
    with (args.output / "sensitivity_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    recommendation = {
        "selection_rule": "minimum geometry ratio score on the three-objective Pareto frontier",
        "pareto_cases": frontier,
        "recommended_case": recommended["case"],
        "normal_weight": recommended["normal_weight"],
        "albedo_window_final_px": recommended["albedo_window"],
        "lunar_lambert_l": recommended["lunar_lambert_l"],
        "pilot_downsample_factor": 4,
        "pilot_iterations_per_level": args.iterations,
        "metrics": {key: recommended[key] for key in objectives},
    }
    (args.output / "recommended_parameters.json").write_text(
        json.dumps(recommendation, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    slope_values = np.array([float(row["stereo_slope20m_deg_rmse"]) for row in rows])
    scatter = axis.scatter(
        [float(row["stereo_elevation_rmse"]) for row in rows],
        [float(row["self_photo_rmse"]) for row in rows],
        c=slope_values, cmap="viridis", s=100, edgecolor="black",
    )
    for row in rows:
        if row["pareto_front"]:
            axis.annotate(str(row["case"]), (float(row["stereo_elevation_rmse"]), float(row["self_photo_rmse"])), fontsize=7)
    figure.colorbar(scatter, ax=axis, label="20 m slope RMSE (deg)")
    axis.set_xlabel("Elevation RMSE vs official stereo DTM (m)")
    axis.set_ylabel("Single-image self photometric RMSE")
    axis.set_title("Wu-2016 SAfS parameter sensitivity (2 m pilot)")
    axis.grid(alpha=0.25)
    figure.savefig(args.output / "parameter_tradeoff.png", dpi=190)
    plt.close(figure)
    print(json.dumps(recommendation, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
