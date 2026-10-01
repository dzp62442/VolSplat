# PandaSet 与 DDAD 使用说明

实现依据：[零样本泛化适配方案](PandaSet与DDAD零样本泛化适配方案.md)。新加载器和评估包装器独立于旧数据集；原 OmniScene 加载、训练、推理实现及配置未修改。

## 数据与划分

`datasets/PandaSet`、`datasets/DDAD` 指向 SVF-GS 使用的同一数据根目录。默认读取各自 `processed`；可用 `dataset.processed_root=/path/to/processed` 显式指定。

- `train`：直接读取 `bins_train.json`。
- `val`：从 `bins_test.json` 用 `linspace` 均匀选择最多 10 个；训练验证保持原 `ValidationWrapper(...,1)` 行为。
- `test`：完整读取 `bins_test.json`，不使用 mini 截断。

所有阶段使用 6 个中心输入、12 个时序新视角及 6 个输入重建视角。RGB、相机参数及深度直接读取 pkl 指定的预处理资产，不运行预处理、不审查数据内容、不按数据质量过滤 bin。

DDAD 的原参考坐标为 `+X 前 / +Y 左 / +Z 上`，加载器将所有相机外参统一左乘 `[[0,-1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]]`，对齐 nuScenes 的 `+X 右 / +Y 前 / +Z 上`。旋转和平移同步变换，相对位姿、长度单位、图像、内参及掩码保持一致；train/val/test 都采用此约定。PandaSet 沿用其原参考坐标。评估摘要通过 `camera_frame` 与 `reference_to_model` 记录 DDAD 的坐标约定。

动态掩码不读取；DDAD 自车掩码只在 val/test 数据加载时读取，用于评估指标。Metric3D-v2 深度仅作为评估 PCC 的参考，不进入 encoder 或训练损失。

## 零样本评估

在项目根目录、`volsplat` 环境下：

```bash
python -m src.main +experiment=pandaset_112x200_zero_shot
python -m src.main +experiment=ddad_112x200_zero_shot
```

默认完整权重：

```text
checkpoints/omniscene-112x200-volsplat-base/checkpoints/epoch_0-step_100000.ckpt
```

实验配置已包含 Base/ViT-B 的 `num_scales=2`、`upsample_factor=2`、`lowest_feature_resolution=4`，无需重复传入。支持通过 `checkpointing.pretrained_model=/path/to/model.ckpt` 更换完整模型权重；编码器和解码器参数必须完整匹配。

224×400 使用相应实验名：

```bash
python -m src.main +experiment=pandaset_224x400_zero_shot
python -m src.main +experiment=ddad_224x400_zero_shot
```

上述四份配置默认均加载现有 112×200 nuScenes 权重。224×400 指目标数据加载和评估尺寸，不代表重新训练了一个 224×400 权重。换用不同来源权重时，同步设置 `source_checkpoint.dataset`、`source_checkpoint.training_resolution`。

DDAD 默认开启自车掩码，也可通过命令行显式指定；结果目录沿用原实验命名：

```bash
python -m src.main +experiment=ddad_112x200_zero_shot \
  dataset.eval_use_ego_mask=true \
  output_dir=outputs/zero_shot/nuscenes_to_ddad_112x200_base_ego_novel12_v1
```

关闭自车掩码时改为 `dataset.eval_use_ego_mask=false`，同时将目录末尾的 `ego_novel12_v1` 改为 `full_image`。自车掩码开关只影响评估像素，两个模式使用相同的数据和相机坐标约定。

## 完整训练配置

以 PandaSet 为例：

```bash
python -m src.main +experiment=pandaset_112x200 \
  checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vitb.pth \
  checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth
```

DDAD 或 224×400 替换对应实验名即可。训练需要 SVF-GS 提供该划分的 `bins_train.json` 及其引用资产，不会自动把测试列表复制成训练列表。

模型、损失、优化器、batch size、100001 步训练上限、每 1% epoch 验证、每 10 次正式验证触发全量 test 等设置均继承对应 OmniScene Base 配置。动态掩码开关置为 false。LPIPS 损失沿用 `apply_after_step=150000`，因此在 100001 步计划内不参与优化；评估 LPIPS 始终正常计算。

DDAD 的快速验证、训练期全量 test 和独立零样本 test 使用相同自车掩码指标。训练损失保持全图。

## 指标、分辨率和输出

| 分组 | 目标索引 | DDAD 支持区域 |
| --- | --- | --- |
| `all_18` | 0–17 | 新视角有效区 + 输入视角全图 |
| `novel_12` | 0–11 | 排除自车区域 |
| `input_6` | 12–17 | 全图 |

PSNR/SSIM/LPIPS 逐视角计算、组内等权平均、跨 bin 等权平均。掩码 SSIM 只使用 11×11 全有效窗口；LPIPS 使用 GT 填充临时预测的无效区，平均 VGG 空间距离图的有效位置。PCC 使用 Metric3D-v2 尺度深度，在每组像素上展平计算，作为诊断字段保存。

遵循原 patch shim：加载 112×200、实际评估 112×192；加载 224×400、实际评估 224×400。RGB、深度、内参、掩码同步处理，不将渲染结果放大回原宽度。

默认目录：

```text
outputs/zero_shot/nuscenes_to_pandaset_<resolution>_base/
outputs/zero_shot/nuscenes_to_ddad_<resolution>_base_ego_novel12_v1/
  resolved_config.yaml
  hydra/<timestamp>/
  metrics/
    per_bin_metrics.csv
    evaluation_summary.json
    scores_all_avg.json
    scores_<metric>_all.json
    per_bin_metrics.rank0000.jsonl
    timing.json
```

`evaluation_summary.json` 的 `groups` 包含三组指标，兼容 `scores_*` 文件仅表示 `all_18`。JSON/CSV 保存完整精度；日志中的 PSNR 保留 3 位，SSIM/LPIPS 保留 4 位。另记录来源权重、配置、加载和实际评估尺寸、像素协议及 PCC 来源。

`complete` 只在全部测试索引的三组记录写出后成立。临时限量运行会留下 `complete=false / limited=true`；不将其作为完整实验结果。进程中途结束时可在各 rank 的 JSONL 中找到已完成记录。未定义指标保留为 null 及原因，不中断、不剔除样本、不静默忽略后重算均值。Infinity 使用明确字符串表示。

训练期全量评估写到 `output_dir/evaluation/step-XXXXXXXX/`；本地可视化写到 `output_dir/local/`，不会清理或覆盖原固定 `outputs/local`。计时记录沿用主机调用边界，明确标注未同步，不作为精确 GPU 延迟报告。

## 调试与验证

临时运行必须同时将结果和 Hydra 日志放在 `/tmp`。新配置的 Hydra 目录及指标目录跟随 `output_dir`：

```bash
python -m src.main +experiment=ddad_112x200_zero_shot \
  output_dir=/tmp/volsplat-ddad-debug
```

上面仍会评估完整 test；单步/单样本调试应在临时测试 harness 中给 Lightning 设置 limit，不能通过改写正式索引实现。

合成测试：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  python -m unittest discover -s tests -p test_temporal18.py -v
```

测试 fixture 固定生成到 `/tmp`，覆盖八套配置一致性、划分和相机顺序、裁剪及掩码对齐、输入视角不变、未定义指标保留、三组聚合和 checkpoint 完整参数匹配。在本机存在 SVF-GS checkout 时，额外直接调用其指标代码进行数值对照。

### 2026-10-01 实施验证记录

- 9 项自动测试通过，包括直接与当前 SVF-GS 掩码指标函数的数值对照。
- RTX 4090 上，PandaSet/DDAD × 112×200/224×400 四组真实样本推理均通过，每组仅处理 1 个 bin；完成 18 视角渲染、三组指标、RGB/深度图及深度数组保存。
- 两个数据集各完成一次合成资产的 GPU 训练步、验证和训练期限量评估。原 `training_step`、优化器及验证计数/调度方法均通过继承使用。
- 两份正式数据根目录的 `processed/bins_train.json` 在实际训练入口读取时均不存在。因此真实训练未执行；没有修改正式数据、复用 test 索引作为 train 或添加自动回退。
- 与开发前工作区逐文件比较，已有 `src/config` 文件仅修改 `src/main.py` 和 `src/dataset/__init__.py`，原有 OmniScene 未提交修改保持一致。

所有调试产物位于 `/tmp/volsplat-temporal18-dev`，其中四份真实样本结果均标记 `complete=false / limited=true`。本轮未运行全量正式评估；测试时临时使用了 6 视角分块渲染，关闭训练验证的投影/视频可视化以缩短调试时间，正式配置未改变这些开关。
