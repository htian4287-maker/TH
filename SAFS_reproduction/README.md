# Wu et al. (2016) 月球 SAfS 独立复现

本目录是对 Wu et al. (2016), *Shape and Albedo from Shading (SAfS) for Pixel-Level DEM Refinement* 的可审计独立复现。它不是作者源代码；论文未公开程序，也没有给出全部数值参数，因此所有缺失参数均写入 YAML 配置并通过合成真值和独立 NAC 立体 DTM 选取。

## 实现内容

- 论文式 (1)：影像强度 = 反照率 × 反射率。
- 式 (2)：四角高程单元的东西、南北坡度。
- 式 (3)：每次二倍上采样后，同一父单元对应的 2×2 子单元法向算术平均约束。
- 式 (4)–(8)：对数域分块常量反照率/反射率分离。
- 式 (9)：四邻域反射率残差与低通法向约束的联合目标。
- 12×11 → 24×22 → … → 1382×1262 的真实论文网格层级。
- 按光照象限的松弛更新；小栅格提供逐节点顺序模式，全 NAC 使用可并行的四色 Gauss–Seidel 模式。
- ISIS 导出的逐像元入射角、发射角、太阳方位角、航天器方位角。
- Lunar-Lambert 反射函数及参数敏感性实验。

## 目录

- `src/safs_method/`：SAfS 核心实现。
- `config/`：合成与 M173 配置。
- `scripts/`：验证、正式重建、参数筛选和三模型对比。
- `tests/`：方程、层级和求解器测试。
- `references/`：论文 PDF、提取文本和逐页渲染图。
- `outputs/`：合成真值验收；真实 NAC 大成果保存在 `E:\NAC_Photometry\paper2016_multi\26_safs_reproduction`。
- `docs/`：算法映射、限制与正式实验报告。

## 运行环境与命令

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nac_photometry
cd '/mnt/e/光度法代码/SAFS方法'

PYTHONPATH=src pytest -q
PYTHONPATH=src python scripts/run_synthetic_validation.py --size 96
PYTHONPATH=src python scripts/run_nac_parameter_sensitivity.py \
  --output /mnt/e/NAC_Photometry/paper2016_multi/26_safs_reproduction/01_m173_2m_sensitivity \
  --iterations 3
PYTHONPATH=src python scripts/run_nac_m173.py \
  --output /mnt/e/NAC_Photometry/paper2016_multi/26_safs_reproduction/02_m173_full/recommended_nw1_aw61_ll0p35 \
  --normal-weight 1.0 --albedo-window 61 --lunar-lambert-l 0.35 --iterations 5
python scripts/compare_single_m173_models.py
```

已经存在 `run-METRICS.json` 的正式案例默认复用，不会重复计算；只有显式增加 `--force` 才覆盖。

## 结论边界

当前结果说明本复现在同一 M173 单景、同一初始 DEM、同一共享水平配准和同一官方 NAC 立体 DTM 下，几何指标优于本次选择的 ASP 单景和 GRUMPE 单景结果。但 SAfS 的高程更新仍存在显著纵向带状结构，且独立 DTM 自身约有 1–2 m 不确定度。结论仅适用于当前小区和当前实现，不能直接外推到 Shackleton 或多景数据集。

