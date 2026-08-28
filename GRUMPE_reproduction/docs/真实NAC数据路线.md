# 从合成验证到真实 NAC 的复现路线

## 两条实验线必须分开

### A. 严格论文复现

两篇 2014 年论文主要使用 M³ 影像、GLD100/LOLA 等数据。若目标是“严格复现论文指标”，应取得论文所列区域和波段，复现其像元尺度、Gaussian 宽度、权重、IMSA/AMSA 和 LOLA 轨道误差统计。

### B. NAC 方法迁移

NAC 是更高分辨率推扫影像。它可以使用相同的数学框架，但传感器、波段、像元尺度、相位覆盖和相机误差都不同，不能沿用 M³ 参数后仍称为严格论文复现。它应被称为“Grumpe 方法在 LRO NAC 上的适配实验”。

## NAC 输入应达到的状态

在进入本工程前，每景影像至少要经过：

1. `lronac2isis`：PDS EDR 转 ISIS Cube；
2. `spiceinit`：附加 SPICE 相机与太阳几何；
3. `lronaccal`：辐射定标；
4. `lronacecho`：回波校正；
5. 多景控制网和 `bundle_adjust`：相机相对几何一致；
6. 初始 DEM 与影像对齐；
7. 用同一 DEM 和 BA 相机执行 `mapproject`；
8. 输出逐像元太阳/观测方向，而不是只使用影像中心角；
9. 所有影像、DEM、掩膜严格同 CRS、分辨率、extent 和 transform；
10. 标记阴影、饱和、NoData、极端入射角和极端发射角。

ISIS/ASP 并不是 Grumpe PHCL/SfS 的唯一实现，但它们目前是 NAC 原始数据预处理和精确相机几何最成熟的工具。核心反射率反演、坡度估计和受约束积分由本工程独立完成。

## 建议的真实数据实验顺序

1. 选择已有 clean8 的小公共重叠窗口，固定所有输入；
2. 做文件、CRS、transform、NoData 和几何符号审计；
3. 只用粗 DEM 法向量拟合每景辐射尺度，画观测/模型散点图；
4. 固定 DHG 参数，只反演逐像元 w；
5. 运行 IMSA + PHCL，保存每景残差；
6. 对 p、q 分别运行可积性检查 `∂p/∂y-∂q/∂x`；
7. 用 Poisson、raw-depth、Grumpe-lowpass 三种方法积分；
8. 用留一景法评价预测残差，避免只比较训练影像；
9. 与原 ASP DEM 比较差值、坡度、方向频谱、条纹指标和独立 LOLA/立体点；
10. 通过小区域后再扩展到完整公共重叠区；
11. IMSA 基线稳定后，才实现并测试 AMSA；
12. 最后建立单景/多景、不同相位跨度、不同权重的完整消融表。

## 数据不放在中文代码目录

按现有项目约束：

- 代码：`E:\光度法代码\GRUMPE方法`；
- 大型数据和结果：`E:\NAC_Photometry\grumpe_reproduction`；
- WSL 对应路径分别为 `/mnt/e/光度法代码/GRUMPE方法` 和 `/mnt/e/NAC_Photometry/grumpe_reproduction`。

