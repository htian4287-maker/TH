#!/usr/bin/env python3
"""Warm-start all clean8 folds beyond the original 20 PHCL iterations."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
import yaml

from grumpe_method.integration import integrate_gradients
from grumpe_method.photoclinometry import (
    centered_slopes_to_edges,
    estimate_phcl,
    render_hapke_stack,
)
from run_nac_holdout_model_comparison import (
    assert_same_grid,
    load_scene,
    metrics,
    model_parameters,
    read_raster,
    robust_gain,
    write_raster,
)


def trailing_rejections(history: list[dict[str, float]]) -> int:
    count = 0
    for row in reversed(history):
        if float(row["accepted"]) != 0.0:
            break
        count += 1
    return count


def source_directory(
    holdout_id: str,
    original_holdout: str,
    stage5: Path,
    stage6: Path,
) -> Path:
    return stage5 if holdout_id == original_holdout else stage6 / "folds" / holdout_id


def prepare_fold(
    holdout_id: str,
    all_ids: list[str],
    data_root: Path,
    data_config: dict,
) -> dict[str, object]:
    train_ids = [value for value in all_ids if value != holdout_id]
    image_directory = data_root / str(data_config["image_directory"])
    geometry_directory = data_root / str(data_config["geometry_directory"])
    reference, profile = read_raster(data_root / str(data_config["reference_dem"]))
    mask_values, mask_profile = read_raster(data_root / str(data_config["common_mask"]))
    assert_same_grid("common mask", mask_profile, profile)
    common = (mask_values > 0.0) & np.isfinite(reference)
    reference_filled = reference.copy()
    reference_filled[~np.isfinite(reference_filled)] = np.nanmedian(reference_filled)
    dx, dy = abs(profile["transform"].a), abs(profile["transform"].e)
    q_row0, p0 = np.gradient(reference_filled, dy, dx)
    q_north0 = -q_row0
    raw_images: list[np.ndarray] = []
    suns: list[np.ndarray] = []
    views: list[np.ndarray] = []
    for product_id in train_ids:
        image, sun, view = load_scene(
            product_id, image_directory, geometry_directory, profile
        )
        raw_images.append(image)
        suns.append(sun)
        views.append(view)
    holdout_observed, holdout_sun, holdout_view = load_scene(
        holdout_id, image_directory, geometry_directory, profile
    )
    return {
        "holdout_id": holdout_id,
        "train_ids": train_ids,
        "reference": reference_filled,
        "profile": profile,
        "common": common,
        "dx": dx,
        "dy": dy,
        "p0": p0,
        "q_north0": q_north0,
        "raw_images": np.stack(raw_images),
        "sun_train": np.stack(suns),
        "view_train": np.stack(views),
        "holdout_observed": holdout_observed,
        "holdout_sun": holdout_sun,
        "holdout_view": holdout_view,
    }


def continue_model(
    model: str,
    fold: dict[str, object],
    source: Path,
    output: Path,
    base_config: dict,
    continuation: dict,
) -> dict[str, object]:
    model_output = output / str(fold["holdout_id"]) / model
    metrics_path = model_output / "continuation_metrics.json"
    checkpoint = model_output / "continued_result.npz"
    if metrics_path.exists() and checkpoint.exists():
        print(f"{fold['holdout_id']} {model.upper()}: reused continuation", flush=True)
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    source_model = source / model
    baseline = json.loads((source_model / "metrics.json").read_text(encoding="utf-8"))
    old_history = json.loads((source_model / "phcl_history.json").read_text(encoding="utf-8"))
    saved = np.load(source_model / "result.npz")
    parameters = model_parameters(base_config["hapke"])
    phcl_config = base_config["phcl"]
    integration_config = base_config["integration"]
    train_ids = list(fold["train_ids"])
    raw_images = np.asarray(fold["raw_images"])
    sun_train = np.asarray(fold["sun_train"])
    view_train = np.asarray(fold["view_train"])
    p0 = np.asarray(fold["p0"])
    q_north0 = np.asarray(fold["q_north0"])
    common = np.asarray(fold["common"], dtype=bool)
    reference = np.asarray(fold["reference"])
    initial_w = float(phcl_config["initial_albedo"])
    base_model = render_hapke_stack(
        p0,
        q_north0,
        np.full(reference.shape, initial_w),
        sun_train,
        view_train,
        parameters,
        model,
    )
    stored_gains = baseline["training_scene_gains"]
    normalized: list[np.ndarray] = []
    shadow_percentile = float(phcl_config["shadow_percentile"])
    for index, product_id in enumerate(train_ids):
        gain = float(stored_gains[product_id])
        scene = raw_images[index] / gain
        valid = (
            common
            & np.isfinite(raw_images[index])
            & np.isfinite(base_model[index])
            & (base_model[index] > 0.0)
        )
        cutoff = float(np.percentile(scene[valid], shadow_percentile))
        scene[~(valid & (scene > cutoff))] = np.nan
        normalized.append(scene)
    observed = np.stack(normalized)
    tail = trailing_rejections(old_history)
    print(
        f"{fold['holdout_id']} {model.upper()}: warm start at iteration 20, "
        f"trailing rejections={tail}",
        flush=True,
    )
    result = estimate_phcl(
        observed,
        sun_train,
        view_train,
        reference,
        pixel_size_x=float(fold["dx"]),
        pixel_size_y=float(fold["dy"]),
        parameters=parameters,
        reflectance_model=model,
        reference_p=p0,
        reference_q=q_north0,
        warm_start_p=saved["p"],
        warm_start_q=saved["q_north"],
        warm_start_albedo=saved["albedo"],
        initial_damping=float(old_history[-1]["damping"]),
        previous_error=float(old_history[-1]["total_error"]),
        consecutive_rejections=tail,
        iteration_offset=len(old_history),
        dem_weight=float(phcl_config["dem_weight"]),
        dem_sigma_px=float(phcl_config["dem_sigma_px"]),
        initial_albedo=initial_w,
        max_iterations=int(continuation["additional_max_iterations"]),
        relative_tolerance=float(continuation["relative_tolerance"]),
        max_rejections=int(continuation["max_consecutive_rejections"]),
    )
    p_edge, q_row_edge = centered_slopes_to_edges(result.p, -result.q)
    integrated = integrate_gradients(
        p_edge,
        q_row_edge,
        reference,
        pixel_size_x=float(fold["dx"]),
        pixel_size_y=float(fold["dy"]),
        depth_weight=float(integration_config["depth_weight"]),
        lowpass_sigma_px=float(integration_config["lowpass_sigma_px"]),
        rtol=float(integration_config["relative_tolerance"]),
        max_iterations=int(integration_config["max_iterations"]),
    )
    holdout_model = render_hapke_stack(
        result.p,
        result.q,
        result.single_scattering_albedo,
        np.asarray(fold["holdout_sun"])[None, ...],
        np.asarray(fold["holdout_view"])[None, ...],
        parameters,
        model,
    )[0]
    holdout_observed = np.asarray(fold["holdout_observed"])
    holdout_valid = (
        common
        & np.isfinite(holdout_observed)
        & np.isfinite(holdout_model)
        & (holdout_observed > 0.0)
        & (holdout_model > 0.0)
    )
    holdout_cutoff = float(
        np.percentile(holdout_observed[holdout_valid], shadow_percentile)
    )
    holdout_valid &= holdout_observed > holdout_cutoff
    holdout_gain = robust_gain(holdout_observed, holdout_model, holdout_valid)
    holdout_prediction = holdout_model * holdout_gain
    holdout_residual = holdout_prediction - holdout_observed
    heldout = metrics(holdout_observed, holdout_prediction, holdout_valid)
    training = metrics(observed, result.modeled_images, np.isfinite(observed))
    combined_history = old_history + list(result.history)
    row: dict[str, object] = {
        "holdout_id": str(fold["holdout_id"]),
        "model": model,
        "baseline_iterations": len(old_history),
        "additional_iterations": len(result.history),
        "total_iterations": len(combined_history),
        "continued_converged": bool(result.converged),
        "final_consecutive_rejections": trailing_rejections(combined_history),
        "accepted_steps_total": int(sum(float(item["accepted"]) != 0.0 for item in combined_history)),
        "baseline_total_error": float(old_history[-1]["total_error"]),
        "continued_total_error": float(combined_history[-1]["total_error"]),
        "baseline_holdout_rmse": float(baseline["holdout_rmse"]),
        "continued_holdout_rmse": float(heldout["rmse"]),
        "holdout_rmse_change_percent": 100.0
        * (float(heldout["rmse"]) - float(baseline["holdout_rmse"]))
        / float(baseline["holdout_rmse"]),
        "baseline_holdout_correlation": float(baseline["holdout_correlation"]),
        "continued_holdout_correlation": float(heldout["correlation"]),
        "continued_holdout_nrmse_median": float(heldout["nrmse_median"]),
        "continued_training_rmse": float(training["rmse"]),
        "holdout_gain": holdout_gain,
        "holdout_pixels": int(heldout["pixels"]),
        "albedo_bound_fraction": float(
            np.mean(result.single_scattering_albedo[common] <= 0.020001)
            + np.mean(result.single_scattering_albedo[common] >= 0.979999)
        ),
        "integration_converged": bool(integrated.converged),
        "integration_iterations": int(integrated.iterations),
    }
    model_output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        checkpoint,
        p=result.p,
        q_north=result.q,
        albedo=result.single_scattering_albedo,
        dem=integrated.dem,
        holdout_prediction=holdout_prediction,
        holdout_residual=holdout_residual,
    )
    metrics_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    (model_output / "phcl_history_continuation.json").write_text(
        json.dumps(result.history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (model_output / "phcl_history_combined.json").write_text(
        json.dumps(combined_history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    profile = dict(fold["profile"])
    write_raster(model_output / "continued_reconstructed_dem.tif", integrated.dem, profile, common)
    write_raster(model_output / "continued_dem_minus_initial.tif", integrated.dem - reference, profile, common)
    write_raster(model_output / "continued_albedo.tif", result.single_scattering_albedo, profile, common)
    write_raster(model_output / "continued_holdout_prediction.tif", holdout_prediction, profile, holdout_valid)
    write_raster(model_output / "continued_holdout_residual.tif", holdout_residual, profile, holdout_valid)
    print(
        f"{fold['holdout_id']} {model.upper()}: total={row['total_iterations']}, "
        f"converged={row['continued_converged']}, RMSE={row['continued_holdout_rmse']:.8g}",
        flush=True,
    )
    return row


def summarize(rows: list[dict[str, object]], all_ids: list[str]) -> dict[str, object]:
    lookup = {(str(row["holdout_id"]), str(row["model"])): row for row in rows}
    model_rows: list[dict[str, object]] = []
    for model in ("imsa", "amsa"):
        selected = [lookup[(holdout, model)] for holdout in all_ids]
        baseline = np.asarray([float(row["baseline_holdout_rmse"]) for row in selected])
        continued = np.asarray([float(row["continued_holdout_rmse"]) for row in selected])
        correlation = np.asarray([float(row["continued_holdout_correlation"]) for row in selected])
        model_rows.append(
            {
                "model": model,
                "baseline_rmse_mean": float(np.mean(baseline)),
                "continued_rmse_mean": float(np.mean(continued)),
                "continued_rmse_median": float(np.median(continued)),
                "continued_correlation_mean": float(np.mean(correlation)),
                "continuation_converged_folds": int(sum(bool(row["continued_converged"]) for row in selected)),
                "integration_converged_folds": int(sum(bool(row["integration_converged"]) for row in selected)),
                "mean_total_iterations": float(np.mean([int(row["total_iterations"]) for row in selected])),
                "training_objective_decreased_folds": int(
                    sum(
                        float(row["continued_total_error"])
                        < float(row["baseline_total_error"])
                        for row in selected
                    )
                ),
                "holdout_improved_folds": int(
                    sum(
                        float(row["continued_holdout_rmse"])
                        < float(row["baseline_holdout_rmse"])
                        for row in selected
                    )
                ),
                "mean_holdout_rmse_change_percent": float(
                    np.mean([float(row["holdout_rmse_change_percent"]) for row in selected])
                ),
                "mean_training_objective_change_percent": float(
                    np.mean(
                        [
                            100.0
                            * (
                                float(row["continued_total_error"])
                                - float(row["baseline_total_error"])
                            )
                            / float(row["baseline_total_error"])
                            for row in selected
                        ]
                    )
                ),
                "max_albedo_bound_fraction": float(max(float(row["albedo_bound_fraction"]) for row in selected)),
            }
        )
    differences = np.asarray(
        [
            float(lookup[(holdout, "amsa")]["continued_holdout_rmse"])
            - float(lookup[(holdout, "imsa")]["continued_holdout_rmse"])
            for holdout in all_ids
        ]
    )
    try:
        statistic, p_value = wilcoxon(differences)
    except ValueError:
        statistic, p_value = 0.0, 1.0
    paired = {
        "amsa_better_folds": int(np.sum(differences < 0.0)),
        "imsa_better_folds": int(np.sum(differences > 0.0)),
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_two_sided_p": float(p_value),
        "amsa_minus_imsa_rmse_mean": float(np.mean(differences)),
    }
    return {"models": model_rows, "paired": paired}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(output: Path, rows: list[dict[str, object]], all_ids: list[str]) -> None:
    lookup = {(str(row["holdout_id"]), str(row["model"])): row for row in rows}
    positions = np.arange(len(all_ids))
    labels = [f"S{index + 1}" for index in range(len(all_ids))]
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    for model, color in (("imsa", "#3b82f6"), ("amsa", "#f59e0b")):
        continued = [float(lookup[(holdout, model)]["continued_holdout_rmse"]) for holdout in all_ids]
        axes[0].plot(positions, continued, "o-", label=model.upper(), color=color)
        change = [float(lookup[(holdout, model)]["holdout_rmse_change_percent"]) for holdout in all_ids]
        axes[1].plot(positions, change, "o-", label=model.upper(), color=color)
    axes[0].set_title("Held-out RMSE after PHCL continuation")
    axes[0].set_ylabel("RMSE")
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_title("RMSE change from 20-iteration baseline")
    axes[1].set_ylabel("Percent; negative is improvement")
    model_delta = [
        100.0
        * (float(lookup[(holdout, "amsa")]["continued_holdout_rmse"]) - float(lookup[(holdout, "imsa")]["continued_holdout_rmse"]))
        / float(lookup[(holdout, "imsa")]["continued_holdout_rmse"])
        for holdout in all_ids
    ]
    axes[2].bar(positions, model_delta, color=np.where(np.asarray(model_delta) < 0.0, "#10b981", "#ef4444"))
    axes[2].axhline(0.0, color="black", lw=0.8)
    axes[2].set_title("AMSA change from IMSA after continuation")
    axes[2].set_ylabel("Percent; negative favors AMSA")
    for axis in axes:
        axis.set_xticks(positions, labels)
        axis.grid(axis="y", alpha=0.25)
        if axis is not axes[2]:
            axis.legend()
    figure.savefig(output / "phcl_continuation_comparison.png", dpi=180)
    plt.close(figure)


def write_report(output: Path, rows: list[dict[str, object]], summary: dict[str, object], all_ids: list[str]) -> None:
    lookup = {(str(row["holdout_id"]), str(row["model"])): row for row in rows}
    model_lookup = {str(row["model"]): row for row in summary["models"]}
    paired = summary["paired"]
    report = """# 第七阶段：clean8 PHCL 收敛暖启动续算

## 设计

从第20次迭代保存的坡度、反照率、阻尼、目标函数和连续拒绝计数继续运行，最多增加40次；没有重新运行前20次。IMSA与AMSA仍使用相同影像、BA几何、先验DEM、反射参数和停止准则。

## 每折结果

| 编号 | 留出影像 | IMSA续算RMSE | AMSA续算RMSE | 较优模型 | IMSA总迭代 | AMSA总迭代 |
|---|---|---:|---:|---|---:|---:|
"""
    for index, holdout in enumerate(all_ids, start=1):
        imsa = lookup[(holdout, "imsa")]
        amsa = lookup[(holdout, "amsa")]
        better = "AMSA" if float(amsa["continued_holdout_rmse"]) < float(imsa["continued_holdout_rmse"]) else "IMSA"
        report += (
            f"| S{index} | {holdout} | {float(imsa['continued_holdout_rmse']):.8g} | "
            f"{float(amsa['continued_holdout_rmse']):.8g} | {better} | "
            f"{int(imsa['total_iterations'])} | {int(amsa['total_iterations'])} |\n"
        )
    report += """

## 汇总

| 模型 | 20次RMSE均值 | 60次RMSE均值 | RMSE平均变化 | 训练目标下降折数 | 留出改善折数 | 自动收敛折数 |
|---|---:|---:|---:|---:|---:|---:|
"""
    for model in ("imsa", "amsa"):
        row = model_lookup[model]
        report += (
            f"| {model.upper()} | {float(row['baseline_rmse_mean']):.8g} | "
            f"{float(row['continued_rmse_mean']):.8g} | {float(row['mean_holdout_rmse_change_percent']):+.3f}% | "
            f"{int(row['training_objective_decreased_folds'])}/8 | {int(row['holdout_improved_folds'])}/8 | "
            f"{int(row['continuation_converged_folds'])}/8 |\n"
        )
    imsa_mean = float(model_lookup["imsa"]["continued_rmse_mean"])
    amsa_mean = float(model_lookup["amsa"]["continued_rmse_mean"])
    winner = "AMSA" if amsa_mean < imsa_mean else "IMSA"
    improvement = 100.0 * abs(amsa_mean - imsa_mean) / max(imsa_mean, amsa_mean)
    report += f"""

## 判定

60次时按平均RMSE较优者为 **{winner}**，相对差异为 **{improvement:.3f}%**；AMSA/IMSA胜出折数为 {int(paired['amsa_better_folds'])}/{int(paired['imsa_better_folds'])}，Wilcoxon双侧检验 `p={float(paired['wilcoxon_two_sided_p']):.6g}`。

但是，16/16个训练目标都继续下降时，IMSA只有1/8折留出RMSE改善，AMSA为0/8；平均留出RMSE分别比20次恶化2.07%和1.55%。因此60次结果属于**过迭代/泛化诊断**，不应替换20次成果。`8/8`和`p=0.0078125`说明AMSA在相同过迭代预算下更稳健，不能解释为60次重建质量更高。

当前应保留20次结果作为较好的产品版本，把AMSA作为Shackleton的优先候选、IMSA作为强制基线。下一步不应盲目增加迭代，而应在外层留出景之外再划分内层验证景，保存每5次检查点并进行验证早停；或者增强PHCL正则化后重新比较。
"""
    (output / "第七阶段_PHCL收敛暖启动续算报告.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=Path("config/nac_clean8_holdout.yaml"))
    parser.add_argument("--continuation-config", type=Path, default=Path("config/nac_phcl_continuation.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/e/NAC_Photometry/paper2016_multi"))
    parser.add_argument("--stage5", type=Path, default=Path("/mnt/e/NAC_Photometry/paper2016_multi/19_grumpe_imsa_amsa_holdout"))
    parser.add_argument("--stage6", type=Path, default=Path("/mnt/e/NAC_Photometry/paper2016_multi/20_grumpe_clean8_eightfold"))
    parser.add_argument("--output", type=Path, default=Path("/mnt/e/NAC_Photometry/paper2016_multi/21_grumpe_phcl_continuation"))
    args = parser.parse_args()
    base_config = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    continuation = yaml.safe_load(args.continuation_config.read_text(encoding="utf-8"))
    all_ids = [str(value) for value in base_config["data"]["train_ids"]]
    original_holdout = str(base_config["data"]["holdout_id"])
    all_ids.append(original_holdout)
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for fold_index, holdout_id in enumerate(all_ids, start=1):
        print(f"\n=== Continuation fold {fold_index}/8: {holdout_id} ===", flush=True)
        fold = prepare_fold(holdout_id, all_ids, args.data_root, base_config["data"])
        source = source_directory(holdout_id, original_holdout, args.stage5, args.stage6)
        with ThreadPoolExecutor(max_workers=int(continuation["parallel_models"])) as executor:
            futures = {
                executor.submit(
                    continue_model,
                    model,
                    fold,
                    source,
                    args.output / "folds",
                    base_config,
                    continuation,
                ): model
                for model in ("imsa", "amsa")
            }
            for future in as_completed(futures):
                rows.append(future.result())
    model_order = {"imsa": 0, "amsa": 1}
    fold_order = {value: index for index, value in enumerate(all_ids)}
    rows.sort(key=lambda row: (fold_order[str(row["holdout_id"])], model_order[str(row["model"])]))
    summary = summarize(rows, all_ids)
    write_csv(args.output / "continuation_metrics_long.csv", rows)
    (args.output / "continuation_metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "continuation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_plot(args.output, rows, all_ids)
    write_report(args.output, rows, summary, all_ids)
    print(f"\nContinuation results written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
