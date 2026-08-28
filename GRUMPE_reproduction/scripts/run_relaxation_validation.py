#!/usr/bin/env python3
"""Validate the paper Eq. (31)--(34) relaxation against reference solvers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from grumpe_method.integration import integrate_gradients
from grumpe_method.photoclinometry import centered_slopes_to_edges, reference_slopes
from grumpe_method.reflectance import HapkeParameters
from grumpe_method.relaxation import (
    gaussian_kernel,
    relaxation_energy,
    solve_paper_relaxation,
)
from grumpe_method.synthetic import make_synthetic_case


def rmse(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - expected) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    case = make_synthetic_case(parameters=HapkeParameters(), **config["synthetic"])
    p, q = reference_slopes(case.truth_dem, case.pixel_size_m, case.pixel_size_m)
    p_biased = p + float(config["stress"]["bias_p"])
    q_biased = q + float(config["stress"]["bias_q"])
    p_edge, q_edge = centered_slopes_to_edges(p_biased, q_biased)
    solver = config["solver"]
    common = dict(pixel_size_x=case.pixel_size_m, pixel_size_y=case.pixel_size_m)
    poisson = integrate_gradients(
        p_edge, q_edge, case.initial_dem,
        depth_weight=0.0, lowpass_sigma_px=0.0, **common,
    )
    raw = integrate_gradients(
        p_edge, q_edge, case.initial_dem,
        depth_weight=float(solver["depth_weight"]), lowpass_sigma_px=0.0, **common,
    )
    cg = integrate_gradients(
        p_edge, q_edge, case.initial_dem,
        depth_weight=float(solver["depth_weight"]),
        lowpass_sigma_px=float(solver["lowpass_sigma_px"]), **common,
    )
    relaxation = solve_paper_relaxation(
        p_biased,
        q_biased,
        case.initial_dem,
        pixel_size_x=case.pixel_size_m,
        pixel_size_y=case.pixel_size_m,
        depth_weight=float(solver["depth_weight"]),
        lowpass_sigma_px=float(solver["lowpass_sigma_px"]),
        max_successful_iterations=int(solver["max_successful_iterations"]),
        max_steps_without_improvement=int(solver["max_steps_without_improvement"]),
        max_total_updates=int(solver["max_total_updates"]),
        relative_tolerance=float(solver["relative_tolerance"]),
    )
    poisson_aligned = poisson.dem + np.mean(case.truth_dem - poisson.dem)
    kernel = gaussian_kernel(float(solver["lowpass_sigma_px"]))
    initial_energy = relaxation_energy(
        case.initial_dem, p_biased, q_biased, case.initial_dem,
        pixel_size_x=case.pixel_size_m, pixel_size_y=case.pixel_size_m,
        depth_weight=float(solver["depth_weight"]), kernel=kernel,
    )[0]
    cg_energy = relaxation_energy(
        cg.dem, p_biased, q_biased, case.initial_dem,
        pixel_size_x=case.pixel_size_m, pixel_size_y=case.pixel_size_m,
        depth_weight=float(solver["depth_weight"]), kernel=kernel,
    )[0]
    metrics = {
        "implementation": "paper Eq. (31)-(34), symmetric Gaussian and constant pixel extents",
        "stress_bias_p": float(config["stress"]["bias_p"]),
        "stress_bias_q": float(config["stress"]["bias_q"]),
        "initial_rmse_m": rmse(case.initial_dem, case.truth_dem),
        "biased_poisson_rmse_m_mean_aligned": rmse(poisson_aligned, case.truth_dem),
        "raw_constraint_rmse_m": rmse(raw.dem, case.truth_dem),
        "cg_reference_rmse_m": rmse(cg.dem, case.truth_dem),
        "paper_relaxation_rmse_m": rmse(relaxation.dem, case.truth_dem),
        "paper_vs_cg_rmse_m": rmse(relaxation.dem, cg.dem),
        "initial_paper_energy": initial_energy,
        "paper_best_energy": relaxation.best_error,
        "cg_evaluated_with_paper_discrete_energy": cg_energy,
        "paper_successful_iterations": relaxation.successful_iterations,
        "paper_total_updates": relaxation.total_updates,
        "paper_steps_since_improvement": relaxation.steps_since_improvement,
        "paper_stop_reason": relaxation.stop_reason,
        "singular_pixels_last_update": relaxation.singular_pixels_last_update,
        "important_note": (
            "Both methods target the continuous Eq. (4), but CG uses forward edge "
            "differences while the published relaxation uses its central-difference "
            "nonlinear Euler/update derivation. Their outputs are compared but are "
            "not expected to be pixel-identical."
        ),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "history.json").write_text(
        json.dumps(relaxation.history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name, values in {
        "truth_dem": case.truth_dem,
        "initial_dem": case.initial_dem,
        "biased_p": p_biased,
        "biased_q": q_biased,
        "poisson_dem_mean_aligned": poisson_aligned,
        "raw_constraint_dem": raw.dem,
        "cg_lowpass_dem": cg.dem,
        "paper_relaxation_dem": relaxation.dem,
        "paper_minus_cg": relaxation.dem - cg.dem,
        "paper_error": relaxation.dem - case.truth_dem,
    }.items():
        np.save(args.output / f"{name}.npy", values)

    panels = [
        ("Truth", case.truth_dem),
        ("Initial coarse DEM", case.initial_dem),
        ("Biased Poisson", poisson_aligned),
        ("Raw constraint", raw.dem),
        ("CG reference", cg.dem),
        ("Published relaxation", relaxation.dem),
        ("Relaxation error", relaxation.dem - case.truth_dem),
        ("Relaxation minus CG", relaxation.dem - cg.dem),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for axis, (title, values) in zip(axes.ravel(), panels):
        image = axis.imshow(values, cmap="coolwarm" if "error" in title.lower() or "minus" in title.lower() else "terrain")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(image, ax=axis, shrink=0.76)
    fig.suptitle("Grumpe & Woehler (2014) published relaxation validation")
    fig.savefig(args.output / "dem_comparison.png", dpi=180)
    plt.close(fig)

    history = relaxation.history
    fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    axis.semilogy([item["update"] for item in history], [item["error"] for item in history])
    axis.axhline(cg_energy, color="tab:red", linestyle="--", label="CG evaluated on paper stencil")
    axis.set_xlabel("Total update step")
    axis.set_ylabel("Eq. (4) discrete energy")
    axis.set_title("Published relaxation convergence")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    fig.savefig(args.output / "convergence.png", dpi=180)
    plt.close(fig)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
