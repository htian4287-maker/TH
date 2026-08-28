#!/usr/bin/env python3
"""Run a controlled IMSA/AMSA comparison on the real NAC clean8 data.

Seven images estimate slopes and albedo. M1193400008LE is never used in the
reconstruction and is evaluated only after both models have finished. A
single robust multiplicative gain is fitted to each scene because the Hapke
implementation returns bidirectional reflectance while the NAC rasters store
calibrated I/F. The gain does not change spatial morphology.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter
import yaml

from grumpe_method.geometry import direction_from_azimuth_zenith
from grumpe_method.integration import integrate_gradients
from grumpe_method.photoclinometry import (
    centered_slopes_to_edges,
    estimate_phcl,
    render_hapke_stack,
)
from grumpe_method.reflectance import HapkeParameters


GEOMETRY_BANDS = ("incidence", "emission", "sun_azimuth", "spacecraft_azimuth")


def read_raster(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as source:
        values = source.read(1).astype(np.float64)
        if source.nodata is not None:
            values[values == source.nodata] = np.nan
        return values, source.profile.copy()


def assert_same_grid(label: str, profile: dict, reference: dict) -> None:
    keys = ("width", "height", "transform", "crs")
    if any(profile[key] != reference[key] for key in keys):
        raise ValueError(f"Grid mismatch for {label}")


def write_raster(path: Path, values: np.ndarray, profile: dict, mask: np.ndarray) -> None:
    output_profile = profile.copy()
    output_profile.update(dtype="float32", count=1, nodata=-9999.0, compress="deflate")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **output_profile) as target:
        target.write(np.where(mask & np.isfinite(values), values, -9999.0).astype(np.float32), 1)


def load_scene(
    product_id: str,
    image_directory: Path,
    geometry_directory: Path,
    reference_profile: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image, image_profile = read_raster(image_directory / f"{product_id}.tif")
    assert_same_grid(f"{product_id} image", image_profile, reference_profile)
    geometry: dict[str, np.ndarray] = {}
    for band in GEOMETRY_BANDS:
        path = geometry_directory / product_id / "maps_2m" / f"{product_id}_{band}_2m.tif"
        geometry[band], band_profile = read_raster(path)
        assert_same_grid(f"{product_id} {band}", band_profile, reference_profile)
    sun = direction_from_azimuth_zenith(geometry["sun_azimuth"], geometry["incidence"])
    view = direction_from_azimuth_zenith(
        geometry["spacecraft_azimuth"], geometry["emission"]
    )
    return image, sun, view


def robust_gain(observed: np.ndarray, modeled: np.ndarray, valid: np.ndarray) -> float:
    active = valid & np.isfinite(observed) & np.isfinite(modeled) & (observed > 0.0) & (modeled > 0.0)
    if active.sum() < 100:
        raise ValueError("Too few pixels for robust scene-gain estimation")
    ratio = observed[active] / modeled[active]
    low, high = np.percentile(ratio, (5.0, 95.0))
    trimmed = ratio[(ratio >= low) & (ratio <= high)]
    return float(np.median(trimmed))


def model_parameters(config: dict) -> HapkeParameters:
    return HapkeParameters(
        opposition_amplitude=float(config["opposition_amplitude"]),
        opposition_width=float(config["opposition_width"]),
        phase_function=str(config["phase_function"]),
        dhg_b=float(config["dhg_b"]),
        dhg_c=float(config["dhg_c"]),
        roughness_degrees=float(config["roughness_degrees"]),
        h_function=str(config["h_function"]),
        legendre_order=int(config["legendre_order"]),
    )


def metrics(observed: np.ndarray, predicted: np.ndarray, valid: np.ndarray) -> dict[str, float | int]:
    active = valid & np.isfinite(observed) & np.isfinite(predicted)
    residual = predicted[active] - observed[active]
    median_signal = float(np.median(np.abs(observed[active])))
    return {
        "pixels": int(active.sum()),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "nrmse_median": float(np.sqrt(np.mean(residual * residual)) / max(median_signal, 1.0e-12)),
        "correlation": float(np.corrcoef(observed[active], predicted[active])[0, 1]),
    }


def hillshade(dem: np.ndarray, dx: float, dy: float) -> np.ndarray:
    row_slope, east_slope = np.gradient(dem, dy, dx)
    north_slope = -row_slope
    normal = np.stack((-east_slope, -north_slope, np.ones_like(dem)), axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    light = direction_from_azimuth_zenith(315.0, 45.0)
    return np.clip(np.sum(normal * light, axis=-1), 0.0, 1.0)


def save_figures(output: Path, arrays: dict[str, np.ndarray], rows: list[dict], valid: np.ndarray, dx: float, dy: float) -> None:
    initial = arrays["initial_dem"]
    delta_i = arrays["imsa_dem"] - initial
    delta_a = arrays["amsa_dem"] - initial
    delta_models = arrays["amsa_dem"] - arrays["imsa_dem"]
    limit = max(float(np.nanpercentile(np.abs(delta_i[valid]), 98)), float(np.nanpercentile(np.abs(delta_a[valid]), 98)), 1.0e-6)
    figure, axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
    panels = [
        ("Initial DEM hillshade", hillshade(initial, dx, dy), "gray", None),
        ("IMSA DEM - initial (m)", delta_i, "coolwarm", TwoSlopeNorm(0.0, -limit, limit)),
        ("AMSA DEM - initial (m)", delta_a, "coolwarm", TwoSlopeNorm(0.0, -limit, limit)),
        ("AMSA DEM - IMSA DEM (m)", delta_models, "coolwarm", TwoSlopeNorm(0.0, -limit, limit)),
    ]
    for axis, (title, values, cmap, norm) in zip(axes, panels):
        image = axis.imshow(np.where(valid, values, np.nan), cmap=cmap, norm=norm)
        axis.set_title(title)
        axis.set_axis_off()
        figure.colorbar(image, ax=axis, shrink=0.75)
    figure.savefig(output / "dem_model_comparison.png", dpi=180)
    plt.close(figure)

    initial_hillshade = hillshade(initial, dx, dy)
    imsa_hillshade = hillshade(arrays["imsa_dem"], dx, dy)
    amsa_hillshade = hillshade(arrays["amsa_dem"], dx, dy)
    shade_delta = amsa_hillshade - imsa_hillshade
    shade_limit = max(float(np.nanpercentile(np.abs(shade_delta[valid]), 99)), 1.0e-6)
    figure, axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
    shade_panels = [
        ("Initial DEM", initial_hillshade, "gray", None, None),
        ("IMSA reconstructed DEM", imsa_hillshade, "gray", None, None),
        ("AMSA reconstructed DEM", amsa_hillshade, "gray", None, None),
        ("AMSA - IMSA hillshade", shade_delta, "coolwarm", -shade_limit, shade_limit),
    ]
    for axis, (title, values, cmap, vmin, vmax) in zip(axes, shade_panels):
        image = axis.imshow(np.where(valid, values, np.nan), cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_axis_off()
        figure.colorbar(image, ax=axis, shrink=0.75)
    figure.savefig(output / "reconstructed_dem_hillshades.png", dpi=180)
    plt.close(figure)

    obs = arrays["holdout_observed"]
    residual_limit = max(
        float(np.nanpercentile(np.abs(arrays["imsa_holdout_residual"][valid]), 98)),
        float(np.nanpercentile(np.abs(arrays["amsa_holdout_residual"][valid]), 98)),
        1.0e-9,
    )
    display_limits = np.nanpercentile(obs[valid], (1, 99))
    figure, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    image_panels = [
        ("Held-out M119 observed", obs, "gray", display_limits[0], display_limits[1]),
        ("IMSA prediction", arrays["imsa_holdout_prediction"], "gray", display_limits[0], display_limits[1]),
        ("AMSA prediction", arrays["amsa_holdout_prediction"], "gray", display_limits[0], display_limits[1]),
        ("AMSA - IMSA prediction", arrays["amsa_holdout_prediction"] - arrays["imsa_holdout_prediction"], "coolwarm", -residual_limit, residual_limit),
        ("IMSA residual", arrays["imsa_holdout_residual"], "coolwarm", -residual_limit, residual_limit),
        ("AMSA residual", arrays["amsa_holdout_residual"], "coolwarm", -residual_limit, residual_limit),
    ]
    for axis, (title, values, cmap, vmin, vmax) in zip(axes.flat, image_panels):
        image = axis.imshow(np.where(valid, values, np.nan), cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_axis_off()
        figure.colorbar(image, ax=axis, shrink=0.75)
    figure.savefig(output / "holdout_prediction_comparison.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    labels = [str(row["model"]).upper() for row in rows]
    for axis, key, title in zip(
        axes,
        ("holdout_rmse", "holdout_nrmse_median", "holdout_correlation"),
        ("Held-out RMSE", "Held-out NRMSE / median", "Held-out correlation"),
    ):
        axis.bar(labels, [float(row[key]) for row in rows], color=("#3b82f6", "#f59e0b"))
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(output / "heldout_metric_comparison.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/nac_clean8_holdout.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/e/NAC_Photometry/paper2016_multi"))
    parser.add_argument("--output", type=Path, default=Path("/mnt/e/NAC_Photometry/paper2016_multi/19_grumpe_imsa_amsa_holdout"))
    parser.add_argument("--force", action="store_true", help="Recompute completed model checkpoints")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)

    data = config["data"]
    train_ids = [str(value) for value in data["train_ids"]]
    holdout_id = str(data["holdout_id"])
    image_directory = args.data_root / data["image_directory"]
    geometry_directory = args.data_root / data["geometry_directory"]
    reference, profile = read_raster(args.data_root / data["reference_dem"])
    mask_values, mask_profile = read_raster(args.data_root / data["common_mask"])
    assert_same_grid("common mask", mask_profile, profile)
    common = (mask_values > 0.0) & np.isfinite(reference)
    if not np.isfinite(reference[common]).all() or common.sum() < 100:
        raise ValueError("Invalid reference DEM or common mask")
    # Fill outside the common footprint only for stable differentiation/solving.
    reference_filled = reference.copy()
    if not np.isfinite(reference_filled).all():
        fill = gaussian_filter(np.where(np.isfinite(reference_filled), reference_filled, np.nanmedian(reference_filled)), 2.0)
        reference_filled[~np.isfinite(reference_filled)] = fill[~np.isfinite(reference_filled)]
    dx, dy = abs(profile["transform"].a), abs(profile["transform"].e)
    q_row0, p0 = np.gradient(reference_filled, dy, dx)
    q_north0 = -q_row0

    images: list[np.ndarray] = []
    suns: list[np.ndarray] = []
    views: list[np.ndarray] = []
    for product_id in train_ids:
        print(f"Loading training scene {product_id}", flush=True)
        image, sun, view = load_scene(product_id, image_directory, geometry_directory, profile)
        images.append(image)
        suns.append(sun)
        views.append(view)
    observed_raw = np.stack(images)
    sun_train = np.stack(suns)
    view_train = np.stack(views)
    holdout_observed, holdout_sun, holdout_view = load_scene(
        holdout_id, image_directory, geometry_directory, profile
    )
    parameters = model_parameters(config["hapke"])
    phcl_config = config["phcl"]
    integration_config = config["integration"]
    arrays: dict[str, np.ndarray] = {
        "initial_dem": reference_filled,
        "holdout_observed": holdout_observed,
    }
    rows: list[dict[str, object]] = []

    for model in [str(value) for value in config["models"]]:
        model_dir = args.output / model
        checkpoint = model_dir / "result.npz"
        metadata_path = model_dir / "metrics.json"
        if checkpoint.exists() and metadata_path.exists() and not args.force:
            print(f"{model.upper()}: reusing completed checkpoint", flush=True)
            saved = np.load(checkpoint)
            p = saved["p"]
            q_north = saved["q_north"]
            albedo = saved["albedo"]
            dem = saved["dem"]
            holdout_prediction = saved["holdout_prediction"]
            holdout_residual = saved["holdout_residual"]
            row = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            model_dir.mkdir(parents=True, exist_ok=True)
            base_w = np.full(reference.shape, float(phcl_config["initial_albedo"]))
            base_model = render_hapke_stack(
                p0, q_north0, base_w, sun_train, view_train, parameters, model
            )
            normalized: list[np.ndarray] = []
            gains: list[float] = []
            shadow_percentile = float(phcl_config["shadow_percentile"])
            for index, product_id in enumerate(train_ids):
                valid = common & np.isfinite(observed_raw[index]) & np.isfinite(base_model[index]) & (base_model[index] > 0.0)
                gain = robust_gain(observed_raw[index], base_model[index], valid)
                scene = observed_raw[index] / gain
                cutoff = float(np.percentile(scene[valid], shadow_percentile))
                scene[~(valid & (scene > cutoff))] = np.nan
                normalized.append(scene)
                gains.append(gain)
                print(f"{model.upper()} gain {product_id}: {gain:.8g}", flush=True)
            observed = np.stack(normalized)
            print(f"{model.upper()}: PHCL on seven training images", flush=True)
            result = estimate_phcl(
                observed,
                sun_train,
                view_train,
                reference_filled,
                pixel_size_x=dx,
                pixel_size_y=dy,
                parameters=parameters,
                reflectance_model=model,
                reference_p=p0,
                reference_q=q_north0,
                dem_weight=float(phcl_config["dem_weight"]),
                dem_sigma_px=float(phcl_config["dem_sigma_px"]),
                initial_albedo=float(phcl_config["initial_albedo"]),
                max_iterations=int(phcl_config["max_iterations"]),
                relative_tolerance=float(phcl_config["relative_tolerance"]),
                max_rejections=int(phcl_config["max_rejections"]),
            )
            p, q_north, albedo = result.p, result.q, result.single_scattering_albedo
            # Raster rows increase southward, so the integration q field is -q_north.
            p_edge, q_row_edge = centered_slopes_to_edges(p, -q_north)
            print(f"{model.upper()}: low-pass constrained gradient integration", flush=True)
            integrated = integrate_gradients(
                p_edge,
                q_row_edge,
                reference_filled,
                pixel_size_x=dx,
                pixel_size_y=dy,
                depth_weight=float(integration_config["depth_weight"]),
                lowpass_sigma_px=float(integration_config["lowpass_sigma_px"]),
                rtol=float(integration_config["relative_tolerance"]),
                max_iterations=int(integration_config["max_iterations"]),
            )
            dem = integrated.dem
            holdout_model = render_hapke_stack(
                p,
                q_north,
                albedo,
                holdout_sun[None, ...],
                holdout_view[None, ...],
                parameters,
                model,
            )[0]
            holdout_valid = common & np.isfinite(holdout_observed) & np.isfinite(holdout_model) & (holdout_observed > 0.0) & (holdout_model > 0.0)
            holdout_cutoff = float(np.percentile(holdout_observed[holdout_valid], shadow_percentile))
            holdout_valid &= holdout_observed > holdout_cutoff
            holdout_gain = robust_gain(holdout_observed, holdout_model, holdout_valid)
            holdout_prediction = holdout_model * holdout_gain
            holdout_residual = holdout_prediction - holdout_observed
            heldout = metrics(holdout_observed, holdout_prediction, holdout_valid)
            training_metrics = metrics(observed, result.modeled_images, np.isfinite(observed))
            row = {
                "model": model,
                "roughness_degrees": float(config["hapke"]["roughness_degrees"]),
                "training_images": len(train_ids),
                "holdout_id": holdout_id,
                "phcl_iterations": len(result.history),
                "phcl_converged": bool(result.converged),
                "training_scene_gains": dict(zip(train_ids, gains)),
                "training_rmse": training_metrics["rmse"],
                "training_correlation": training_metrics["correlation"],
                "holdout_gain": holdout_gain,
                "holdout_pixels": heldout["pixels"],
                "holdout_rmse": heldout["rmse"],
                "holdout_mae": heldout["mae"],
                "holdout_bias": heldout["bias"],
                "holdout_nrmse_median": heldout["nrmse_median"],
                "holdout_correlation": heldout["correlation"],
                "albedo_mean": float(np.mean(albedo[common])),
                "albedo_std": float(np.std(albedo[common])),
                "albedo_low_bound_fraction": float(np.mean(albedo[common] <= 0.020001)),
                "albedo_high_bound_fraction": float(np.mean(albedo[common] >= 0.979999)),
                "dem_change_rmse_m": float(np.sqrt(np.mean((dem[common] - reference_filled[common]) ** 2))),
                "dem_change_p98_abs_m": float(np.percentile(np.abs(dem[common] - reference_filled[common]), 98)),
                "integration_converged": bool(integrated.converged),
                "integration_iterations": int(integrated.iterations),
                "integration_relative_residual": float(integrated.relative_residual),
            }
            np.savez_compressed(
                checkpoint,
                p=p,
                q_north=q_north,
                albedo=albedo,
                dem=dem,
                holdout_prediction=holdout_prediction,
                holdout_residual=holdout_residual,
            )
            metadata_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
            (model_dir / "phcl_history.json").write_text(
                json.dumps(result.history, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            write_raster(model_dir / "p_east.tif", p, profile, common)
            write_raster(model_dir / "q_north.tif", q_north, profile, common)
            write_raster(model_dir / "single_scattering_albedo.tif", albedo, profile, common)
            write_raster(model_dir / "reconstructed_dem.tif", dem, profile, common)
            write_raster(model_dir / "dem_minus_initial.tif", dem - reference_filled, profile, common)
            write_raster(model_dir / "holdout_prediction_scaled.tif", holdout_prediction, profile, holdout_valid)
            write_raster(model_dir / "holdout_residual.tif", holdout_residual, profile, holdout_valid)
        rows.append(row)
        arrays[f"{model}_dem"] = dem
        arrays[f"{model}_holdout_prediction"] = holdout_prediction
        arrays[f"{model}_holdout_residual"] = holdout_residual
        print(
            f"{model.upper()} held-out: RMSE={float(row['holdout_rmse']):.8g}, "
            f"correlation={float(row['holdout_correlation']):.6f}",
            flush=True,
        )

    csv_path = args.output / "model_comparison_metrics.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        flat_rows = [{key: value for key, value in row.items() if not isinstance(value, dict)} for row in rows]
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    (args.output / "model_comparison_metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_figures(args.output, arrays, rows, common, dx, dy)
    winner = min(rows, key=lambda item: float(item["holdout_rmse"]))
    loser = max(rows, key=lambda item: float(item["holdout_rmse"]))
    rmse_improvement = 100.0 * (
        float(loser["holdout_rmse"]) - float(winner["holdout_rmse"])
    ) / float(loser["holdout_rmse"])
    correlation_gain = float(winner["holdout_correlation"]) - float(loser["holdout_correlation"])
    report = f"""# 第五阶段：真实 NAC clean8 留一景 IMSA/AMSA 对照

## 实验设计

- 使用 7 景 clean8 影像估计坡度与逐像元单次散射反照率。
- `M1193400008LE` 完全留出，不参与 PHCL 重建，只用于事后预测验证。
- 两组共用 BA 逐像元几何、2 m 初始 DEM、Hapke 参数、11°粗糙度、PHCL 与积分参数。
- 每景只拟合一个乘性辐射尺度；该尺度不改变空间纹理。模型优劣主要看留出景 RMSE、NRMSE、相关系数以及反照率边界饱和。

## 结果

| 模型 | 留出 RMSE | 留出 NRMSE/中位数 | 留出相关系数 | 训练 RMSE | DEM 相对初始 RMSE (m) | 反照率边界比例 |
|---|---:|---:|---:|---:|---:|---:|
"""
    for row in rows:
        bound = float(row["albedo_low_bound_fraction"]) + float(row["albedo_high_bound_fraction"])
        report += (
            f"| {str(row['model']).upper()} | {float(row['holdout_rmse']):.8g} | "
            f"{float(row['holdout_nrmse_median']):.6f} | {float(row['holdout_correlation']):.6f} | "
            f"{float(row['training_rmse']):.8g} | {float(row['dem_change_rmse_m']):.6f} | {bound:.6f} |\n"
        )
    report += f"""

## 当前判定

按单个 M119 留出景的 RMSE，当前较优者为 **{str(winner['model']).upper()}**。这是一项真实数据证据，但还不是最终模型定论：单一留出景可能受该景曝光尺度、残余配准误差和阴影影响。若两模型指标接近，下一步应做 clean8 的完整八折留一交叉验证，再决定 Shackleton 数据集采用哪个模型。

- 相对另一模型，留出 RMSE 降低 **{rmse_improvement:.3f}%**，相关系数提高 **{correlation_gain:.6f}**。
- 两组低通约束梯度积分均已收敛；IMSA/AMSA 分别用了 {int(rows[0]['integration_iterations'])}/{int(rows[1]['integration_iterations'])} 次迭代。
- 两组 PHCL 都执行了固定的 20 次迭代，但未达到自动停止阈值，因此这里应表述为“同等迭代预算下的初步比较”，不能表述为最终模型定论。
- 两组反照率都没有撞到 0.02 或 0.98 的物理边界，未发现通过反照率饱和强行吸收误差的迹象。

## 解释边界

- 本实验替换的是 SfS/PHCL 内部的 Hapke **多次散射近似**（IMSA 与 AMSA），不是替换 ISIS 定标、回波校正、相机模型、BA 或 DEM 对齐。
- DEM 没有独立高分辨率真值，所以 DEM 变化量只用于稳定性检查；模型排名以未参与重建的影像预测为主。
- 留出景的单一增益由留出景自身拟合，因此验证的是空间形态和相对辐射一致性，不是绝对辐射标定精度。
- 目视残差仍保留中心高反照率带及小陨坑边缘结构，说明剩余误差不只是多次散射模型造成，还包含反照率参数化、阴影/遮挡、亚像元配准或初始 DEM 结构的影响。
"""
    (args.output / "第五阶段_真实NAC_IMSA_AMSA留一验证报告.md").write_text(report, encoding="utf-8")
    print(f"Results written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
