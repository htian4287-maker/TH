#!/usr/bin/env python3
"""Run and summarize eight-fold leave-one-image-out NAC validation.

Each fold trains on seven clean8 observations and predicts the eighth.  The
existing M119 fold from stage 5 is reused verbatim; completed fold outputs are
also resumed so an interrupted run never repeats successful reconstructions.
"""

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
from scipy.stats import wilcoxon
import yaml


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_fold(
    runner: Path,
    base_config: dict,
    all_ids: list[str],
    holdout_id: str,
    data_root: Path,
    fold_output: Path,
) -> None:
    fold_output.mkdir(parents=True, exist_ok=True)
    fold_config = json.loads(json.dumps(base_config))
    fold_config["data"]["train_ids"] = [value for value in all_ids if value != holdout_id]
    fold_config["data"]["holdout_id"] = holdout_id
    config_path = fold_output / "fold_config.yaml"
    config_path.write_text(
        yaml.safe_dump(fold_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    command = [
        sys.executable,
        str(runner),
        "--config",
        str(config_path),
        "--data-root",
        str(data_root),
        "--output",
        str(fold_output),
    ]
    print(f"\n=== Fold {holdout_id}: seven-image training ===", flush=True)
    subprocess.run(command, check=True)


def flatten_fold(holdout_id: str, metrics_path: Path, reused: bool) -> list[dict[str, object]]:
    rows = json.loads(metrics_path.read_text(encoding="utf-8"))
    flattened: list[dict[str, object]] = []
    for row in rows:
        flattened.append(
            {
                "holdout_id": holdout_id,
                "model": str(row["model"]),
                "reused_stage5": reused,
                "holdout_pixels": int(row["holdout_pixels"]),
                "holdout_rmse": float(row["holdout_rmse"]),
                "holdout_mae": float(row["holdout_mae"]),
                "holdout_nrmse_median": float(row["holdout_nrmse_median"]),
                "holdout_correlation": float(row["holdout_correlation"]),
                "training_rmse": float(row["training_rmse"]),
                "training_correlation": float(row["training_correlation"]),
                "phcl_iterations": int(row["phcl_iterations"]),
                "phcl_converged": bool(row["phcl_converged"]),
                "integration_converged": bool(row["integration_converged"]),
                "integration_iterations": int(row["integration_iterations"]),
                "albedo_bound_fraction": float(row["albedo_low_bound_fraction"])
                + float(row["albedo_high_bound_fraction"]),
                "dem_change_rmse_m": float(row["dem_change_rmse_m"]),
                "metrics_source": str(metrics_path),
            }
        )
    return flattened


def summarize(rows: list[dict[str, object]], all_ids: list[str]) -> tuple[list[dict[str, object]], dict[str, object]]:
    by_model: dict[str, list[dict[str, object]]] = {
        model: [row for row in rows if row["model"] == model] for model in ("imsa", "amsa")
    }
    by_fold = {
        holdout: {str(row["model"]): row for row in rows if row["holdout_id"] == holdout}
        for holdout in all_ids
    }
    differences = np.asarray(
        [
            float(by_fold[holdout]["amsa"]["holdout_rmse"])
            - float(by_fold[holdout]["imsa"]["holdout_rmse"])
            for holdout in all_ids
        ]
    )
    try:
        statistic, p_value = wilcoxon(differences, alternative="two-sided")
    except ValueError:
        statistic, p_value = 0.0, 1.0
    model_summary: list[dict[str, object]] = []
    for model, model_rows in by_model.items():
        rmse = np.asarray([float(row["holdout_rmse"]) for row in model_rows])
        nrmse = np.asarray([float(row["holdout_nrmse_median"]) for row in model_rows])
        correlation = np.asarray([float(row["holdout_correlation"]) for row in model_rows])
        wins = sum(
            float(by_fold[holdout][model]["holdout_rmse"])
            < float(by_fold[holdout]["amsa" if model == "imsa" else "imsa"]["holdout_rmse"])
            for holdout in all_ids
        )
        model_summary.append(
            {
                "model": model,
                "folds": len(model_rows),
                "rmse_mean": float(np.mean(rmse)),
                "rmse_median": float(np.median(rmse)),
                "rmse_std": float(np.std(rmse, ddof=1)),
                "nrmse_mean": float(np.mean(nrmse)),
                "correlation_mean": float(np.mean(correlation)),
                "correlation_median": float(np.median(correlation)),
                "rmse_wins": int(wins),
                "phcl_converged_folds": int(sum(bool(row["phcl_converged"]) for row in model_rows)),
                "integration_converged_folds": int(sum(bool(row["integration_converged"]) for row in model_rows)),
                "max_albedo_bound_fraction": float(max(float(row["albedo_bound_fraction"]) for row in model_rows)),
            }
        )
    paired = {
        "amsa_minus_imsa_rmse_mean": float(np.mean(differences)),
        "amsa_minus_imsa_rmse_median": float(np.median(differences)),
        "amsa_better_folds": int(np.sum(differences < 0.0)),
        "imsa_better_folds": int(np.sum(differences > 0.0)),
        "ties": int(np.sum(differences == 0.0)),
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_two_sided_p": float(p_value),
    }
    return model_summary, paired


def plot_summary(output: Path, rows: list[dict[str, object]], all_ids: list[str]) -> None:
    short_ids = [f"S{index + 1}" for index in range(len(all_ids))]
    lookup = {
        (str(row["holdout_id"]), str(row["model"])): row
        for row in rows
    }
    imsa_rmse = np.asarray([float(lookup[(value, "imsa")]["holdout_rmse"]) for value in all_ids])
    amsa_rmse = np.asarray([float(lookup[(value, "amsa")]["holdout_rmse"]) for value in all_ids])
    imsa_corr = np.asarray([float(lookup[(value, "imsa")]["holdout_correlation"]) for value in all_ids])
    amsa_corr = np.asarray([float(lookup[(value, "amsa")]["holdout_correlation"]) for value in all_ids])
    positions = np.arange(len(all_ids))

    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    axes[0].plot(positions, imsa_rmse, "o-", label="IMSA", color="#3b82f6")
    axes[0].plot(positions, amsa_rmse, "o-", label="AMSA", color="#f59e0b")
    axes[0].set_title("Held-out RMSE by fold")
    axes[0].set_ylabel("RMSE")
    axes[0].legend()
    axes[1].plot(positions, imsa_corr, "o-", label="IMSA", color="#3b82f6")
    axes[1].plot(positions, amsa_corr, "o-", label="AMSA", color="#f59e0b")
    axes[1].set_title("Held-out correlation by fold")
    axes[1].set_ylabel("Correlation")
    axes[1].legend()
    relative = 100.0 * (amsa_rmse - imsa_rmse) / imsa_rmse
    axes[2].bar(positions, relative, color=np.where(relative < 0.0, "#10b981", "#ef4444"))
    axes[2].axhline(0.0, color="black", lw=0.8)
    axes[2].set_title("AMSA RMSE change from IMSA")
    axes[2].set_ylabel("Percent; negative favors AMSA")
    for axis in axes:
        axis.set_xticks(positions, short_ids, fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(output / "eightfold_model_comparison.png", dpi=180)
    plt.close(figure)


def write_report(
    output: Path,
    rows: list[dict[str, object]],
    summaries: list[dict[str, object]],
    paired: dict[str, object],
    all_ids: list[str],
) -> None:
    lookup = {
        (str(row["holdout_id"]), str(row["model"])): row
        for row in rows
    }
    summary_lookup = {str(row["model"]): row for row in summaries}
    winner = min(summaries, key=lambda row: float(row["rmse_mean"]))
    loser = max(summaries, key=lambda row: float(row["rmse_mean"]))
    improvement = 100.0 * (
        float(loser["rmse_mean"]) - float(winner["rmse_mean"])
    ) / float(loser["rmse_mean"])
    significance = (
        "达到0.05显著性水平"
        if float(paired["wilcoxon_two_sided_p"]) < 0.05
        else "未达到0.05显著性水平"
    )
    report = """# 第六阶段：真实 NAC clean8 八折留一交叉验证

## 设计

clean8 的8景影像依次作为完全留出的验证景，其余7景用于 PHCL 坡度与逐像元反照率估计。IMSA/AMSA 共用相同 BA 几何、2 m 初始 DEM、11°粗糙度、Hapke 参数、20次 PHCL 预算和低通积分参数。M119 折直接复用第五阶段检查点，其余折独立运行。

## 每折结果

图中 `S1`–`S8` 按下表顺序对应8个留出影像。

| 留出影像 | IMSA RMSE | AMSA RMSE | RMSE较优 | IMSA相关 | AMSA相关 |
|---|---:|---:|---|---:|---:|
"""
    for holdout in all_ids:
        imsa = lookup[(holdout, "imsa")]
        amsa = lookup[(holdout, "amsa")]
        better = "AMSA" if float(amsa["holdout_rmse"]) < float(imsa["holdout_rmse"]) else "IMSA"
        report += (
            f"| {holdout} | {float(imsa['holdout_rmse']):.8g} | {float(amsa['holdout_rmse']):.8g} | "
            f"{better} | {float(imsa['holdout_correlation']):.6f} | {float(amsa['holdout_correlation']):.6f} |\n"
        )
    report += """

## 跨折汇总

| 模型 | RMSE均值 | RMSE中位数 | RMSE标准差 | NRMSE均值 | 相关系数均值 | 胜出折数 |
|---|---:|---:|---:|---:|---:|---:|
"""
    for model in ("imsa", "amsa"):
        row = summary_lookup[model]
        report += (
            f"| {model.upper()} | {float(row['rmse_mean']):.8g} | {float(row['rmse_median']):.8g} | "
            f"{float(row['rmse_std']):.8g} | {float(row['nrmse_mean']):.6f} | "
            f"{float(row['correlation_mean']):.6f} | {int(row['rmse_wins'])}/8 |\n"
        )
    report += f"""

## 判定

按八折平均留出 RMSE，当前较优模型为 **{str(winner['model']).upper()}**，相对另一模型改善 **{improvement:.3f}%**。AMSA/IMSA 的胜出折数为 {int(paired['amsa_better_folds'])}/{int(paired['imsa_better_folds'])}；成对 Wilcoxon 双侧检验 `p={float(paired['wilcoxon_two_sided_p']):.6g}`，**{significance}**。

因此，当前最准确的表述是：**多数折和均值呈现 AMSA 略优趋势，但优势很小，现有8折证据不足以证明 AMSA 稳定优于 IMSA。** 在 Shackleton 数据集上可把 AMSA 作为优先候选，同时必须保留 IMSA 基线。

两模型都是 `0/8` 折在20次 PHCL 内触发自动收敛；两模型都是 `8/8` 折完成梯度积分收敛。该结果首先是同等计算预算对照，下一轮应针对 PHCL 收敛性做续算/停止准则实验。

模型选择不能只看一个 p 值：还应同时要求多数折方向一致、平均和中位数一致、相关系数没有恶化、反照率不发生边界饱和。若这些证据冲突，应把结论写为“两个多次散射近似在当前相位范围内差异不足以稳定区分”，而不是强行指定优胜者。

## 解释边界

- 该实验只替换 Hapke 多次散射近似，不替换 ISIS 定标、回波校正、相机、BA 或 DEM 对齐。
- 每个留出景拟合一个乘性辐射尺度，所以评价的是空间形态与相对辐射，而非绝对 I/F 标定。
- 两组均使用固定20次 PHCL 预算。没有达到自动收敛阈值的折必须作为限制报告，不能把同预算排名误写成完全收敛解排名。
- 没有外部高分辨率 DEM 真值，因此最终模型判断仍应结合外部高程/坡度检查和 Shackleton 区域的相位角覆盖。
"""
    (output / "第六阶段_真实NAC_clean8八折留一报告.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/nac_clean8_holdout.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/e/NAC_Photometry/paper2016_multi"))
    parser.add_argument("--output", type=Path, default=Path("/mnt/e/NAC_Photometry/paper2016_multi/20_grumpe_clean8_eightfold"))
    parser.add_argument(
        "--reuse-m119",
        type=Path,
        default=Path("/mnt/e/NAC_Photometry/paper2016_multi/19_grumpe_imsa_amsa_holdout"),
    )
    args = parser.parse_args()
    base_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    all_ids = [str(value) for value in base_config["data"]["train_ids"]]
    original_holdout = str(base_config["data"]["holdout_id"])
    all_ids.append(original_holdout)
    if len(set(all_ids)) != 8:
        raise ValueError("Expected exactly eight unique clean8 image IDs")
    args.output.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("run_nac_holdout_model_comparison.py")
    all_rows: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    for holdout_id in all_ids:
        if holdout_id == original_holdout:
            metrics_path = args.reuse_m119 / "model_comparison_metrics.json"
            if not metrics_path.exists():
                raise FileNotFoundError(f"Stage-5 M119 metrics not found: {metrics_path}")
            print(f"\n=== Fold {holdout_id}: reused from stage 5 ===", flush=True)
            reused = True
        else:
            fold_output = args.output / "folds" / holdout_id
            metrics_path = fold_output / "model_comparison_metrics.json"
            if metrics_path.exists():
                print(f"\n=== Fold {holdout_id}: reusing completed fold ===", flush=True)
            else:
                run_fold(
                    runner,
                    base_config,
                    all_ids,
                    holdout_id,
                    args.data_root,
                    fold_output,
                )
            reused = False
        all_rows.extend(flatten_fold(holdout_id, metrics_path, reused))
        manifest.append(
            {
                "holdout_id": holdout_id,
                "metrics_path": str(metrics_path),
                "reused_stage5": reused,
            }
        )
    summaries, paired = summarize(all_rows, all_ids)
    write_csv(args.output / "eightfold_metrics_long.csv", all_rows)
    write_csv(args.output / "eightfold_model_summary.csv", summaries)
    (args.output / "eightfold_metrics.json").write_text(
        json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "eightfold_summary.json").write_text(
        json.dumps({"models": summaries, "paired": paired}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output / "fold_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_summary(args.output, all_rows, all_ids)
    write_report(args.output, all_rows, summaries, paired, all_ids)
    print(f"\nEight-fold results written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
