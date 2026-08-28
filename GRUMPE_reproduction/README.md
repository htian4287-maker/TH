# GRUMPE 方法复现工程

本工程复现下列两篇论文提出的月球光度地形恢复框架：

1. Grumpe, Belkhir & Wöhler (2014), *Construction of lunar DEMs based on reflectance modelling*.
2. Grumpe & Wöhler (2014), *Recovery of elevation from estimated gradient fields constrained by digital elevation maps of lower lateral resolution*.

## 当前里程碑

当前版本是一个**可运行的分阶段复现**，独立实现：

- 坡度、表面法向量、太阳与观测方向之间的几何关系；
- Hapke IMSA 反射率、阴影隐藏冲日效应、DHG 与 Cornette–Shanks 单粒子相函数；
- Hapke 2002 AMSA Legendre 多次散射，以及 Hapke 1984 宏观粗糙度有效角与阴影修正；
- 论文式多景 PHCL 目标：影像反射率误差加粗 DEM 的低通坡度约束；
- 逐像元反照率与坡度的交替 Levenberg–Marquardt / Gauss–Newton 优化；
- Horn/Poisson、逐像元高程约束以及 Grumpe 低通绝对高程约束三种梯度积分对照；
- 第二篇论文 Eq. (31)–(34) 在对称 Gaussian、固定像元尺寸条件下的逐像元松弛更新与论文停止准则；
- 论文范围 `τ=10⁻⁵…10⁷`、`σ=1,3,…,15 px` 的104组参数筛选、松弛候选复核及 PHCL→高程端到端验证；
- 已知真值的合成地形端到端验证、指标与图件输出。
- IMSA/AMSA、11°/0°粗糙度的7组严格消融，以及5个噪声种子、20次完整稳健性复核。
- 真实 NAC clean8 的7景训练、M119留一验证，并输出 IMSA/AMSA 的预测残差、反照率与重建 DEM 对照。
- 真实 NAC clean8 的完整8折留一交叉验证、成对统计与逐折检查点。
- 从第20次优化器状态严格暖启动到60次的收敛/泛化诊断，确认继续下降训练目标会造成留出过拟合。

尚未宣称完成的部分明确列在 [复现状态](docs/复现状态.md) 中，尤其包括 B_CB、论文的光照无关配准、扩展 SfS、变分重建，以及 M³/NAC 真实数据上的参数复现。

## 为什么先做合成验证

真实 NAC 影像没有逐像元地形真值。若直接运行真实数据，只能看到“生成了一张 DEM”，无法证明反射率、坡度优化和高程积分是否正确。合成实验提供真实 DEM、真实坡度和真实反照率，可分别计算 RMSE，定位误差来自哪一模块。

## 在 WSL Ubuntu 中安装

```bash
cd '/mnt/e/光度法代码/GRUMPE方法'
source ~/miniconda3/etc/profile.d/conda.sh
conda env create -f environment.yml
conda activate grumpe_method
python -m pip install -e .
```

如果 `grumpe_method` 环境已经存在：

```bash
conda activate grumpe_method
python -m pip install -e .
```

## 运行单元测试

```bash
cd '/mnt/e/光度法代码/GRUMPE方法'
conda activate grumpe_method
pytest -q
```

## 运行合成端到端复现

```bash
python scripts/run_synthetic_validation.py \
  --config config/synthetic.yaml \
  --output outputs/synthetic_imsa
```

运行第二篇论文的严格松弛对照：

```bash
python scripts/run_relaxation_validation.py \
  --config config/relaxation_validation.yaml \
  --output outputs/relaxation_validation
```

运行论文参数网格与 PHCL 端到端实验：

```bash
python scripts/run_parameter_grid.py \
  --config config/parameter_grid.yaml \
  --output outputs/parameter_grid
```

若104组筛选和候选松弛已经完成，只继续后半段：

```bash
python scripts/run_parameter_grid.py \
  --config config/parameter_grid.yaml \
  --output outputs/parameter_grid \
  --resume
```

运行 IMSA/AMSA 与粗糙度严格对照：

```bash
python scripts/run_reflectance_model_comparison.py \
  --config config/reflectance_model_comparison.yaml \
  --output outputs/reflectance_model_comparison

python scripts/run_reflectance_seed_sweep.py \
  --config config/reflectance_model_comparison.yaml \
  --output outputs/reflectance_model_comparison
```

若只需补算配置中新加入的实验，可在第一条命令后加 `--resume`。

主要输出包括：

- `truth_dem.npy`：合成真值 DEM；
- `initial_dem.npy`：模糊的低分辨率先验 DEM；
- `images.npy`：按多组照明几何渲染的 Hapke 合成影像；
- `estimated_p.npy`、`estimated_q.npy`：PHCL 估计坡度；
- `estimated_w.npy`：逐像元单次散射反照率；
- `poisson_dem.npy`：只积分坡度的结果；
- `raw_depth_dem.npy`：直接约束先验 DEM 的结果；
- `grumpe_lowpass_dem.npy`：仅用先验 DEM 低频成分约束的结果；
- `metrics.json`：各阶段 RMSE 和收敛信息；
- `comparison.png`：真值、先验、三种积分结果及差值图。

## 真实 NAC 数据

合成验证通过后，再按 [真实 NAC 数据路线](docs/真实NAC数据路线.md) 接入现有 ISIS/ASP 预处理成果。ISIS/ASP 负责辐射定标、回波校正、SPICE 相机几何、BA 和投影；本工程负责论文的反射率—坡度—高程核心，不把 ASP `sfs` 当作 Grumpe 算法本身。

当前 clean8 留一验证可直接复用已完成的影像、BA 几何和初始 DEM：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nac_photometry
cd '/mnt/e/光度法代码/GRUMPE方法'
export PYTHONPATH=src
python scripts/run_nac_holdout_model_comparison.py \
  --config config/nac_clean8_holdout.yaml \
  --data-root /mnt/e/NAC_Photometry/paper2016_multi \
  --output /mnt/e/NAC_Photometry/paper2016_multi/19_grumpe_imsa_amsa_holdout
```

脚本默认复用已经成功写入的模型检查点，不会重复计算；只有显式加入 `--force` 才会重算。

运行完整 clean8 八折留一验证：

```bash
python scripts/run_nac_clean8_cross_validation.py \
  --config config/nac_clean8_holdout.yaml \
  --data-root /mnt/e/NAC_Photometry/paper2016_multi \
  --output /mnt/e/NAC_Photometry/paper2016_multi/20_grumpe_clean8_eightfold \
  --reuse-m119 /mnt/e/NAC_Photometry/paper2016_multi/19_grumpe_imsa_amsa_holdout
```

编排脚本会复用已有的M119和各折检查点。当前八折结果为：AMSA在6/8折胜出，平均RMSE改善0.951%，但Wilcoxon `p=0.25`，只能视为AMSA略优趋势，不能视为统计上已经确认。

从第20次状态继续进行PHCL收敛诊断：

```bash
python scripts/run_nac_phcl_continuation.py \
  --base-config config/nac_clean8_holdout.yaml \
  --continuation-config config/nac_phcl_continuation.yaml \
  --data-root /mnt/e/NAC_Photometry/paper2016_multi \
  --stage5 /mnt/e/NAC_Photometry/paper2016_multi/19_grumpe_imsa_amsa_holdout \
  --stage6 /mnt/e/NAC_Photometry/paper2016_multi/20_grumpe_clean8_eightfold \
  --output /mnt/e/NAC_Photometry/paper2016_multi/21_grumpe_phcl_continuation
```

续算脚本继承坡度、反照率、阻尼、目标函数和连续拒绝计数，不重新运行前20次。真实clean8结果显示：训练目标继续下降，但60次留出RMSE比20次更差，因此20次成果仍是当前较好的产品版本；60次只作为过拟合诊断。
