#!/usr/bin/env python3
"""Paper-range tau/sigma search followed by PHCL integration validation."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from grumpe_method.integration import integrate_gradients
from grumpe_method.photoclinometry import (
    centered_slopes_to_edges,
    estimate_phcl,
    reference_slopes,
)
from grumpe_method.reflectance import HapkeParameters
from grumpe_method.relaxation import solve_paper_relaxation
from grumpe_method.synthetic import make_synthetic_case


def _rmse(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - expected) ** 2)))


def _mean_aligned_rmse(actual: np.ndarray, expected: np.ndarray) -> float:
    aligned = actual + np.mean(expected - actual)
    return _rmse(aligned, expected)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _relaxation_worker(payload: dict[str, object]) -> dict[str, object]:
    result = solve_paper_relaxation(
        payload["p"],
        payload["q"],
        payload["initial"],
        pixel_size_x=float(payload["pixel_size"]),
        pixel_size_y=float(payload["pixel_size"]),
        depth_weight=float(payload["tau"]),
        lowpass_sigma_px=float(payload["sigma"]),
        max_successful_iterations=int(payload["max_successful_iterations"]),
        max_steps_without_improvement=int(payload["max_steps_without_improvement"]),
        max_total_updates=int(payload["max_total_updates"]),
        relative_tolerance=float(payload["relative_tolerance"]),
    )
    return {
        "tau_exponent": int(payload["tau_exponent"]),
        "tau": float(payload["tau"]),
        "sigma_px": int(payload["sigma"]),
        "cg_screen_rmse_m": float(payload["cg_screen_rmse_m"]),
        "relaxation_rmse_m": _rmse(result.dem, payload["truth"]),
        "best_energy": result.best_error,
        "successful_iterations": result.successful_iterations,
        "total_updates": result.total_updates,
        "stop_reason": result.stop_reason,
        "singular_pixels": result.singular_pixels_last_update,
    }


def _paper_solve(
    p: np.ndarray,
    q: np.ndarray,
    initial: np.ndarray,
    truth: np.ndarray,
    *,
    pixel_size: float,
    tau: float,
    sigma: float,
    settings: dict[str, object],
) -> tuple[np.ndarray, dict[str, object]]:
    result = solve_paper_relaxation(
        p,
        q,
        initial,
        pixel_size_x=pixel_size,
        pixel_size_y=pixel_size,
        depth_weight=tau,
        lowpass_sigma_px=sigma,
        max_successful_iterations=int(settings["max_successful_iterations"]),
        max_steps_without_improvement=int(settings["max_steps_without_improvement"]),
        max_total_updates=int(settings["max_total_updates"]),
        relative_tolerance=float(settings["relative_tolerance"]),
    )
    return result.dem, {
        "rmse_m": _rmse(result.dem, truth),
        "best_energy": result.best_error,
        "successful_iterations": result.successful_iterations,
        "total_updates": result.total_updates,
        "stop_reason": result.stop_reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed grid/finalist tables and continue with PHCL validation",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    grid = config["grid"]
    relaxation_settings = config["relaxation"]
    parameters = HapkeParameters()

    # Phase A: full paper-range screen using the fast reference discretization.
    screen_case = make_synthetic_case(
        size=int(grid["size"]),
        pixel_size_m=float(grid["pixel_size_m"]),
        prior_sigma_px=float(grid["prior_sigma_px"]),
        noise_std=0.0,
        seed=42,
        parameters=parameters,
    )
    p_truth, q_truth = reference_slopes(
        screen_case.truth_dem, screen_case.pixel_size_m, screen_case.pixel_size_m
    )
    p_screen = p_truth + float(grid["bias_p"])
    q_screen = q_truth + float(grid["bias_q"])
    p_edge, q_edge = centered_slopes_to_edges(p_screen, q_screen)
    cg_rows: list[dict[str, object]] = []
    total = len(grid["tau_exponents"]) * len(grid["sigma_px"])
    completed = 0
    grid_csv = args.output / "cg_full_grid.csv"
    if args.resume and grid_csv.is_file():
        with grid_csv.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                cg_rows.append(
                    {
                        "tau_exponent": int(row["tau_exponent"]),
                        "tau": float(row["tau"]),
                        "sigma_px": int(row["sigma_px"]),
                        "rmse_m": float(row["rmse_m"]),
                        "converged": row["converged"].lower() == "true",
                        "iterations": int(row["iterations"]),
                        "relative_residual": float(row["relative_residual"]),
                    }
                )
        print(f"Resumed {len(cg_rows)} completed CG grid rows", flush=True)
    else:
        for exponent in grid["tau_exponents"]:
            tau = 10.0 ** int(exponent)
            for sigma in grid["sigma_px"]:
                result = integrate_gradients(
                    p_edge,
                    q_edge,
                    screen_case.initial_dem,
                    pixel_size_x=screen_case.pixel_size_m,
                    pixel_size_y=screen_case.pixel_size_m,
                    depth_weight=tau,
                    lowpass_sigma_px=float(sigma),
                    rtol=1.0e-8,
                    max_iterations=3000,
                )
                completed += 1
                row = {
                    "tau_exponent": int(exponent),
                    "tau": tau,
                    "sigma_px": int(sigma),
                    "rmse_m": _rmse(result.dem, screen_case.truth_dem),
                    "converged": result.converged,
                    "iterations": result.iterations,
                    "relative_residual": result.relative_residual,
                }
                cg_rows.append(row)
                print(f"CG screen {completed}/{total}: tau=1e{exponent}, sigma={sigma}, RMSE={row['rmse_m']:.6f}", flush=True)
        _write_csv(grid_csv, cg_rows)
    eligible = [row for row in cg_rows if row["converged"] and np.isfinite(row["rmse_m"])]
    eligible.sort(key=lambda item: float(item["rmse_m"]))
    finalists = eligible[: int(grid["relaxation_finalists"])]

    # Phase B: published relaxation for the CG-screen finalists.
    payloads: list[dict[str, object]] = []
    for row in finalists:
        payloads.append(
            {
                "p": p_screen,
                "q": q_screen,
                "initial": screen_case.initial_dem,
                "truth": screen_case.truth_dem,
                "pixel_size": screen_case.pixel_size_m,
                "tau_exponent": row["tau_exponent"],
                "tau": row["tau"],
                "sigma": row["sigma_px"],
                "cg_screen_rmse_m": row["rmse_m"],
                **relaxation_settings,
            }
        )
    relaxation_rows: list[dict[str, object]] = []
    finalists_csv = args.output / "relaxation_finalists.csv"
    if args.resume and finalists_csv.is_file() and (args.output / "best_parameters.json").is_file():
        with finalists_csv.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                relaxation_rows.append(
                    {
                        "tau_exponent": int(row["tau_exponent"]),
                        "tau": float(row["tau"]),
                        "sigma_px": int(row["sigma_px"]),
                        "cg_screen_rmse_m": float(row["cg_screen_rmse_m"]),
                        "relaxation_rmse_m": float(row["relaxation_rmse_m"]),
                        "best_energy": float(row["best_energy"]),
                        "successful_iterations": int(row["successful_iterations"]),
                        "total_updates": int(row["total_updates"]),
                        "stop_reason": row["stop_reason"],
                        "singular_pixels": int(row["singular_pixels"]),
                    }
                )
        print(f"Resumed {len(relaxation_rows)} completed relaxation finalists", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=int(grid["workers"])) as pool:
            futures = {pool.submit(_relaxation_worker, payload): payload for payload in payloads}
            for future in as_completed(futures):
                payload = futures[future]
                try:
                    row = future.result()
                except Exception as error:  # noqa: BLE001 - preserve failed parameter set
                    row = {
                        "tau_exponent": payload["tau_exponent"],
                        "tau": payload["tau"],
                        "sigma_px": payload["sigma"],
                        "cg_screen_rmse_m": payload["cg_screen_rmse_m"],
                        "relaxation_rmse_m": float("inf"),
                        "best_energy": float("inf"),
                        "successful_iterations": 0,
                        "total_updates": 0,
                        "stop_reason": f"error: {type(error).__name__}: {error}",
                        "singular_pixels": -1,
                    }
                relaxation_rows.append(row)
                print(
                    "Relaxation finalist: "
                    f"tau=1e{row['tau_exponent']}, sigma={row['sigma_px']}, "
                    f"RMSE={row['relaxation_rmse_m']}",
                    flush=True,
                )
        relaxation_rows.sort(key=lambda item: float(item["relaxation_rmse_m"]))
        _write_csv(finalists_csv, relaxation_rows)
    relaxation_rows.sort(key=lambda item: float(item["relaxation_rmse_m"]))
    finite_finalists = [row for row in relaxation_rows if np.isfinite(row["relaxation_rmse_m"])]
    if not finite_finalists:
        raise RuntimeError("All published-relaxation finalists failed")
    best = finite_finalists[0]
    (args.output / "best_parameters.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Phase C: new multi-image PHCL case, then raw and declared-biased integration.
    phcl_cfg = config["phcl_validation"]
    full_case = make_synthetic_case(
        size=int(phcl_cfg["size"]),
        pixel_size_m=float(phcl_cfg["pixel_size_m"]),
        prior_sigma_px=float(phcl_cfg["prior_sigma_px"]),
        noise_std=float(phcl_cfg["noise_std"]),
        seed=int(phcl_cfg["seed"]),
        parameters=parameters,
    )
    phcl = estimate_phcl(
        full_case.images,
        full_case.sun_directions,
        full_case.view_directions,
        full_case.initial_dem,
        pixel_size_x=full_case.pixel_size_m,
        pixel_size_y=full_case.pixel_size_m,
        parameters=parameters,
        dem_weight=float(phcl_cfg["dem_weight"]),
        dem_sigma_px=float(phcl_cfg["dem_sigma_px"]),
        max_iterations=int(phcl_cfg["max_iterations"]),
    )
    raw_p, raw_q = phcl.p, phcl.q
    stressed_p = raw_p + float(grid["bias_p"])
    stressed_q = raw_q + float(grid["bias_q"])
    selected_tau = float(best["tau"])
    selected_sigma = float(best["sigma_px"])
    raw_paper, raw_paper_meta = _paper_solve(
        raw_p, raw_q, full_case.initial_dem, full_case.truth_dem,
        pixel_size=full_case.pixel_size_m, tau=selected_tau,
        sigma=selected_sigma, settings=relaxation_settings,
    )
    stressed_paper, stressed_paper_meta = _paper_solve(
        stressed_p, stressed_q, full_case.initial_dem, full_case.truth_dem,
        pixel_size=full_case.pixel_size_m, tau=selected_tau,
        sigma=selected_sigma, settings=relaxation_settings,
    )
    raw_p_edge, raw_q_edge = centered_slopes_to_edges(raw_p, raw_q)
    stressed_p_edge, stressed_q_edge = centered_slopes_to_edges(stressed_p, stressed_q)
    raw_poisson = integrate_gradients(
        raw_p_edge, raw_q_edge, full_case.initial_dem,
        pixel_size_x=full_case.pixel_size_m, pixel_size_y=full_case.pixel_size_m,
        depth_weight=0.0, lowpass_sigma_px=0.0,
    ).dem
    stressed_poisson = integrate_gradients(
        stressed_p_edge, stressed_q_edge, full_case.initial_dem,
        pixel_size_x=full_case.pixel_size_m, pixel_size_y=full_case.pixel_size_m,
        depth_weight=0.0, lowpass_sigma_px=0.0,
    ).dem
    phcl_metrics = {
        "selected_tau": selected_tau,
        "selected_tau_exponent": int(best["tau_exponent"]),
        "selected_sigma_px": selected_sigma,
        "phcl_converged": phcl.converged,
        "phcl_iterations": len(phcl.history),
        "phcl_p_rmse": _rmse(raw_p, full_case.truth_p),
        "phcl_q_rmse": _rmse(raw_q, full_case.truth_q),
        "initial_dem_rmse_m": _rmse(full_case.initial_dem, full_case.truth_dem),
        "raw_phcl_poisson_rmse_m_mean_aligned": _mean_aligned_rmse(raw_poisson, full_case.truth_dem),
        "raw_phcl_paper_relaxation": raw_paper_meta,
        "stress_bias_p": float(grid["bias_p"]),
        "stress_bias_q": float(grid["bias_q"]),
        "stressed_phcl_poisson_rmse_m_mean_aligned": _mean_aligned_rmse(stressed_poisson, full_case.truth_dem),
        "stressed_phcl_paper_relaxation": stressed_paper_meta,
    }
    (args.output / "phcl_integration_metrics.json").write_text(
        json.dumps(phcl_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name, values in {
        "phcl_truth_dem": full_case.truth_dem,
        "phcl_initial_dem": full_case.initial_dem,
        "phcl_estimated_p": raw_p,
        "phcl_estimated_q": raw_q,
        "raw_phcl_poisson": raw_poisson + np.mean(full_case.truth_dem - raw_poisson),
        "raw_phcl_paper": raw_paper,
        "stressed_phcl_poisson": stressed_poisson + np.mean(full_case.truth_dem - stressed_poisson),
        "stressed_phcl_paper": stressed_paper,
    }.items():
        np.save(args.output / f"{name}.npy", values)

    # Figures.
    exponents = [int(value) for value in grid["tau_exponents"]]
    sigmas = [int(value) for value in grid["sigma_px"]]
    heat = np.full((len(exponents), len(sigmas)), np.nan)
    for row in cg_rows:
        heat[exponents.index(int(row["tau_exponent"])), sigmas.index(int(row["sigma_px"]))] = float(row["rmse_m"])
    fig, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    image = axis.imshow(heat, aspect="auto", origin="lower", cmap="viridis")
    axis.set_xticks(range(len(sigmas)), sigmas)
    axis.set_yticks(range(len(exponents)), [f"1e{value}" for value in exponents])
    axis.set_xlabel("Gaussian sigma (px)")
    axis.set_ylabel("Depth weight tau")
    axis.set_title("Full 104-combination CG screening RMSE (m)")
    fig.colorbar(image, ax=axis, label="DEM RMSE (m)")
    fig.savefig(args.output / "cg_grid_heatmap.png", dpi=180)
    plt.close(fig)

    panels = [
        ("Truth", full_case.truth_dem),
        ("Initial DEM", full_case.initial_dem),
        ("Raw PHCL Poisson", raw_poisson + np.mean(full_case.truth_dem - raw_poisson)),
        ("Raw PHCL paper relaxation", raw_paper),
        ("Stressed PHCL Poisson", stressed_poisson + np.mean(full_case.truth_dem - stressed_poisson)),
        ("Stressed PHCL paper relaxation", stressed_paper),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for axis, (title, values) in zip(axes.ravel(), panels):
        shown = axis.imshow(values, cmap="terrain")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(shown, ax=axis, shrink=0.75)
    fig.suptitle("Selected paper parameters applied to PHCL output")
    fig.savefig(args.output / "phcl_dem_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"best_parameters": best, "phcl_metrics": phcl_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
