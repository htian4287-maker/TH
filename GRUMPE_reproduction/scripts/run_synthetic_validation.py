#!/usr/bin/env python3
"""Run the synthetic Grumpe PHCL and constrained-integration experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from grumpe_method.integration import integrate_gradients
from grumpe_method.photoclinometry import centered_slopes_to_edges, estimate_phcl
from grumpe_method.reflectance import HapkeParameters
from grumpe_method.synthetic import make_synthetic_case


def rmse(actual: np.ndarray, expected: np.ndarray, mask: np.ndarray | None = None) -> float:
    valid = np.isfinite(actual) & np.isfinite(expected)
    if mask is not None:
        valid &= mask
    return float(np.sqrt(np.mean((actual[valid] - expected[valid]) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)

    hapke_cfg = config["hapke"]
    hapke = HapkeParameters(
        single_scattering_albedo=0.4,
        opposition_amplitude=float(hapke_cfg["opposition_amplitude"]),
        opposition_width=float(hapke_cfg["opposition_width"]),
        phase_function=str(hapke_cfg["phase_function"]),
        dhg_b=float(hapke_cfg["dhg_b"]),
        dhg_c=float(hapke_cfg["dhg_c"]),
        cornette_shanks_n=float(hapke_cfg["cornette_shanks_n"]),
    )
    case = make_synthetic_case(parameters=hapke, **config["synthetic"])
    phcl_cfg = config["phcl"]
    phcl = estimate_phcl(
        case.images,
        case.sun_directions,
        case.view_directions,
        case.initial_dem,
        pixel_size_x=case.pixel_size_m,
        pixel_size_y=case.pixel_size_m,
        parameters=hapke,
        dem_weight=float(phcl_cfg["dem_weight"]),
        dem_sigma_px=float(phcl_cfg["dem_sigma_px"]),
        initial_albedo=float(phcl_cfg["initial_albedo"]),
        max_iterations=int(phcl_cfg["max_iterations"]),
        relative_tolerance=float(phcl_cfg["relative_tolerance"]),
    )
    p_edge, q_edge = centered_slopes_to_edges(phcl.p, phcl.q)
    integration_cfg = config["integration"]
    common = dict(
        pixel_size_x=case.pixel_size_m,
        pixel_size_y=case.pixel_size_m,
        rtol=float(integration_cfg["rtol"]),
        max_iterations=int(integration_cfg["max_iterations"]),
    )
    clean_poisson = integrate_gradients(
        p_edge, q_edge, case.initial_dem,
        depth_weight=0.0, lowpass_sigma_px=0.0, **common
    )
    # Grumpe & Woehler (2014) target accumulated low-frequency height drift
    # caused by a small systematic gradient error.  Apply a declared constant
    # slope bias as a controlled stress test; never confuse it with PHCL output.
    biased_p_edge = p_edge + float(integration_cfg["stress_bias_p"])
    biased_q_edge = q_edge + float(integration_cfg["stress_bias_q"])
    poisson = integrate_gradients(
        biased_p_edge, biased_q_edge, case.initial_dem,
        depth_weight=0.0, lowpass_sigma_px=0.0, **common
    )
    raw = integrate_gradients(
        biased_p_edge, biased_q_edge, case.initial_dem,
        depth_weight=float(integration_cfg["depth_weight"]),
        lowpass_sigma_px=0.0, **common
    )
    grumpe = integrate_gradients(
        biased_p_edge, biased_q_edge, case.initial_dem,
        depth_weight=float(integration_cfg["depth_weight"]),
        lowpass_sigma_px=float(integration_cfg["lowpass_sigma_px"]), **common
    )

    arrays = {
        "truth_dem": case.truth_dem,
        "initial_dem": case.initial_dem,
        "truth_p": case.truth_p,
        "truth_q": case.truth_q,
        "truth_w": case.truth_w,
        "images": case.images,
        "sun_directions": case.sun_directions,
        "view_directions": case.view_directions,
        "estimated_p": phcl.p,
        "estimated_q": phcl.q,
        "estimated_w": phcl.single_scattering_albedo,
        "modeled_images": phcl.modeled_images,
        "valid_mask": phcl.valid_mask,
        "phcl_p_edge": p_edge,
        "phcl_q_edge": q_edge,
        "stress_biased_p_edge": biased_p_edge,
        "stress_biased_q_edge": biased_q_edge,
        "clean_poisson_dem": clean_poisson.dem,
        "poisson_dem": poisson.dem,
        "raw_depth_dem": raw.dem,
        "grumpe_lowpass_dem": grumpe.dem,
    }
    for name, values in arrays.items():
        np.save(args.output / f"{name}.npy", values)

    # Poisson has an arbitrary vertical datum; align means for fair RMSE.
    clean_poisson_aligned = clean_poisson.dem + np.mean(
        case.truth_dem - clean_poisson.dem
    )
    poisson_aligned = poisson.dem + np.mean(case.truth_dem - poisson.dem)
    metrics = {
        "initial_dem_rmse_m": rmse(case.initial_dem, case.truth_dem),
        "phcl_p_rmse": rmse(phcl.p, case.truth_p, phcl.valid_mask),
        "phcl_q_rmse": rmse(phcl.q, case.truth_q, phcl.valid_mask),
        "albedo_rmse": rmse(
            phcl.single_scattering_albedo, case.truth_w, phcl.valid_mask
        ),
        "clean_poisson_dem_rmse_m_mean_aligned": rmse(
            clean_poisson_aligned, case.truth_dem
        ),
        "stress_bias_p": float(integration_cfg["stress_bias_p"]),
        "stress_bias_q": float(integration_cfg["stress_bias_q"]),
        "biased_poisson_dem_rmse_m_mean_aligned": rmse(
            poisson_aligned, case.truth_dem
        ),
        "raw_depth_dem_rmse_m": rmse(raw.dem, case.truth_dem),
        "grumpe_lowpass_dem_rmse_m": rmse(grumpe.dem, case.truth_dem),
        "phcl_converged": phcl.converged,
        "phcl_iterations": len(phcl.history),
        "integration": {
            "clean_poisson": asdict(clean_poisson) | {"dem": "clean_poisson_dem.npy"},
            "biased_poisson": asdict(poisson) | {"dem": "poisson_dem.npy"},
            "raw_depth": asdict(raw) | {"dem": "raw_depth_dem.npy"},
            "grumpe_lowpass": asdict(grumpe) | {"dem": "grumpe_lowpass_dem.npy"},
        },
        "note": "Poisson RMSE is computed after mean-elevation alignment.",
    }
    # Replace any remaining arrays from dataclass serialization defensively.
    for value in metrics["integration"].values():
        value.pop("dem", None)
    metrics["integration"]["clean_poisson"]["file"] = "clean_poisson_dem.npy"
    metrics["integration"]["biased_poisson"]["file"] = "poisson_dem.npy"
    metrics["integration"]["raw_depth"]["file"] = "raw_depth_dem.npy"
    metrics["integration"]["grumpe_lowpass"]["file"] = "grumpe_lowpass_dem.npy"
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "phcl_history.json").write_text(
        json.dumps(phcl.history, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    surfaces = [
        ("Truth DEM", case.truth_dem),
        ("Low-resolution prior", case.initial_dem),
        ("Biased Poisson (mean aligned)", poisson_aligned),
        ("Raw depth constraint", raw.dem),
        ("Grumpe low-pass constraint", grumpe.dem),
        ("Grumpe error", grumpe.dem - case.truth_dem),
        ("True albedo w", case.truth_w),
        ("Estimated albedo w", phcl.single_scattering_albedo),
        ("Albedo error", phcl.single_scattering_albedo - case.truth_w),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(14, 12), constrained_layout=True)
    for axis, (title, values) in zip(axes.ravel(), surfaces):
        image = axis.imshow(values, cmap="terrain" if "DEM" in title or "constraint" in title else "coolwarm")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(image, ax=axis, shrink=0.78)
    fig.suptitle("Grumpe synthetic PHCL + constrained integration validation")
    fig.savefig(args.output / "comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
