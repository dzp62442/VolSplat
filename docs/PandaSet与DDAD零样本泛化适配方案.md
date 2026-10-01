# PandaSet 与 DDAD 零样本泛化适配方案

日期：2026-10-01。状态：方案阶段，尚未实现加载器或运行训练、推理。

本方案依据 VolSplat `comp_svfgs` 当前工作区和 SVF-GS 当前源代码制定。目标是用 OmniScene 格式 nuScenes 上训练的 VolSplat 权重，在 PandaSet、DDAD 的既有 temporal18 预处理数据上评估，同时提供与本项目 OmniScene 设置一致的完整训练配置。

## 1. 范围与已确认决策

| 事项 | 决策 |
| --- | --- |
| 预训练来源 | OmniScene 格式 nuScenes；本地已有 Base/ViT-B、step 100000 权重作为默认零样本来源 |
| 配置覆盖 | PandaSet、DDAD 均提供 112×200、224×400 两档训练及零样本评估配置 |
| 当前默认评估档位 | 与现有权重及 README 测试命令对应的 112×200 Base 配置；224×400 配置一并提供，不在本轮启动实验 |
| 数据资产 | 直接读取 SVF-GS 已生成的索引、RGB、内参、位姿、Metric3D-v2 深度，以及 DDAD 自车掩码 |
| 划分 | 仅 `train / val / test`；分别对应 SVF-GS 当前加载器的同名划分 |
| 视角 | 中心 6 输入；18 目标 = 前 12 时序新视角 + 后 6 输入视角 |
| 动态物体 | 全部视为静态；不生成、不读取动态掩码，训练关闭动态掩码开关 |
| DDAD 自车掩码 | 所有评估阶段的 novel_12 使用；input_6 保持全图；训练损失不用 |
| PCC | 使用既有 Metric3D-v2 尺度深度计算并保留诊断值，正式指标展示以 PSNR、SSIM、LPIPS 为主 |
| 分辨率行为 | 完全沿用 OmniScene patch shim；加载 112×200 后实际评估 112×192，224×400 不变；结果记录两个尺寸 |
| 原有实现 | 不修改已有数据集加载器、模型、损失、优化器、原评估包装器及原配置；只新增适配代码和必要入口分支 |
| 数据信任边界 | 不审查数据内容，不重新筛选样本，不设置数据质量、哈希、范围、身份一致性等中断条件 |

本轮经用户授权已建立：

```text
datasets/PandaSet -> /home/dzp62442/Projects/SVF-GS/data/PandaSet
datasets/DDAD     -> /home/dzp62442/Projects/SVF-GS/data/DDAD
```

只建立链接，没有修改链接目标中的数据，也没有遍历、校验数据集内容。

工作区原有 `src/dataset/dataset_omniscene.py` 未提交修改：测试索引从 mini 切换为全量。该修改属于既有工作，原样保留。

## 2. 当前实现与接入边界

### 2.1 VolSplat OmniScene 路径

```text
src/main.py
  -> Hydra 配置 / typed config
  -> DataModule -> get_dataset -> DatasetOmniScene
  -> ModelWrapper
      -> encoder.get_data_shim() -> VolSplat encoder -> splatting_cuda decoder
      -> training_step / validation_step / test_step
      -> on_validation_epoch_end -> run_full_test_sets_eval
```

当前 OmniScene 数据集输出 `context / target / scene`，分别含 6 输入和 18 目标。相机外参直接使用 `sensor2lidar_transform`，内参按图像宽高归一化。测试时以 DepthAnything-v2 相对深度计算 PCC；训练动态掩码逻辑位于原 `ModelWrapper.training_step`。

现有测试指标只整体汇总全部目标视角，没有独立的 `all_18 / novel_12 / input_6` 报告。新需求的分组和自车掩码指标应在新包装器实现，不能顺带改变 OmniScene 的报告或计算路径。

`src/main.py` 的训练期评估配置构建只专门识别 `omniscene`，其他分支按 re10k/scannet 查 evaluation index。接入新数据集时，必须让 PandaSet/DDAD 走固定 temporal18 视角路径，避免触发 `Fail to load eval index path`。无需添加 re10k 风格的 evaluation index。

### 2.2 SVF-GS 当前参考路径

当前 `PandaSetDataset`、`DDADDataset` 均继承 `data/temporal_dataset.py::TemporalDataset`，`only_input` 默认是 `False`。本方案采用当前 temporal18 实现，不使用历史上只有输入 6 视角的协议。

参考内容分为两部分：

- 迁移：划分规则、相机顺序、资产路径读取、RGB/深度 resize、相机坐标含义、DDAD 掩码映射、指标公式。
- 不迁移：`validate_protocol`、selection/manifest/bin/RGB/depth 哈希检查、相机和深度数值审查、全资产启动扫描、原始轨迹筛选、预处理和掩码重建。

新加载器在本项目内独立实现，不在运行时导入 SVF-GS 的模型、配置系统或带数据审查的加载器。

## 3. 数据接口与索引

### 3.1 根路径及资产读取

数据集配置分别设置 `roots: [datasets/PandaSet]`、`roots: [datasets/DDAD]`，默认 `processed_root` 为对应根路径下的 `processed`；允许显式覆盖以适配部署路径。

直接读取：

```text
processed/
  bins_train.json
  bins_test.json
  bin_infos/<bin_token>.pkl
  <sensor.data_path>
  <sensor.intrinsic_path>
  <sensor.depth_path>                  # 计算 PCC 时读取
  ego_masks/vidar_v1/manifest.json      # 仅 DDAD 评估
```

`sensor` 内的资产路径相对于 `processed_root` 解析，不根据相机名称或时间戳重新猜测文件名。当前 SVF-GS 生成路径通常使用 `images_small`、`params_small`，深度分别位于 PandaSet 的 `dptm`、DDAD 的 `dptm_small`；加载器以 pkl 指定路径为准。

VolSplat encoder 不消费 Metric3D 深度或置信度。Metric3D 深度仅供评估 PCC，不作为模型输入、训练监督或测试时优化信号；无需为本适配读取置信度及深度身份元数据。测试无需启动 Metric3D、DepthAnything 的数据预处理服务。

PandaSet/DDAD 加载不以数据集 manifest、selection 文件或其中的 `complete`、校验和等字段为前置条件。DDAD 掩码 manifest 是查找掩码文件的映射表，需要读取，但不承担数据合规或质量审查功能。

### 3.2 train / val / test 对应

| VolSplat stage | SVF-GS 对应 | 索引与选择方式 |
| --- | --- | --- |
| `train` | `split="train"` | `bins_train.json["bins"]` 原序列全量加载；训练 DataLoader 按原逻辑 shuffle |
| `val` | `split="val"` | 从 `bins_test.json["bins"]` 按 `np.linspace(0, N-1, min(10, N), dtype=int)` 选择 |
| `test` | `split="test"`，即其 `test_split=total` | `bins_test.json["bins"]` 全量加载，保持顺序 |

不增加 `mini / demo / center150` 等新划分，不复用 OmniScene 的 `0::14` 或按 3000 间隔抽样规则，不自行重新划分 train/test，也不对索引去重或过滤。

需要区分数据集划分与训练验证调度：`val` Dataset 包含上述最多 10 个 bin；原 VolSplat `DataModule.val_dataloader` 使用 `ValidationWrapper(dataset, 1)`，每次验证从中随机取 1 个 bin。为保持 OmniScene 行为，继续使用这一包装器；每 10 次正式验证后的全量评估则遍历 `test`。

### 3.3 相机与目标顺序

| 语义顺序 | pkl 逻辑相机键 | PandaSet 物理相机 | DDAD 物理相机 |
| --- | --- | --- | --- |
| 0 前 | `CAM_FRONT` | `front_camera` | `CAMERA_01` |
| 1 右前 | `CAM_FRONT_RIGHT` | `front_right_camera` | `CAMERA_06` |
| 2 左前 | `CAM_FRONT_LEFT` | `front_left_camera` | `CAMERA_05` |
| 3 后 | `CAM_BACK` | `back_camera` | `CAMERA_09` |
| 4 左后 | `CAM_BACK_LEFT` | `left_camera` | `CAMERA_07` |
| 5 右后 | `CAM_BACK_RIGHT` | `right_camera` | `CAMERA_08` |

每个相机的记录已有 `[center, before, after]`：

```text
context: cam0.center, cam1.center, ..., cam5.center
target:  cam0.before, cam0.after, ..., cam5.before, cam5.after,
         cam0.center, cam1.center, ..., cam5.center
```

不重新按距离选择 before/after，也不以复制中心图像的方式凑满 18 个目标。无动态掩码的要求不改变上述新视角定义。

### 3.4 相机几何与 RGB

1. RGB 直接读取已处理图像，转为 `[0,1]` 的 float32 CHW；按 SVF-GS temporal loader 使用 PIL BILINEAR 调整到加载分辨率。
2. `camera_intrinsic` 从 `intrinsic_path` 读取，按已处理 RGB 的尺寸与加载尺寸缩放，再将第 0 行除以加载宽度、第 1 行除以加载高度，得到 VolSplat 的归一化 K。
3. `extrinsics` 读取 pkl 的 `sensor2lidar_transform`。SVF-GS 生成该矩阵时已经把每个图像时刻的相机转换到中心 LiDAR 坐标，不重算相对位姿，不重置相机平移，不进行单位基线归一化。PandaSet 直接沿用。DDAD 在加载器内统一左乘 `A=[[0,-1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]]`，将原参考系的前/左/上转换为 nuScenes 的右/前/上；输入、目标及 train/val/test 全部一致，旋转和平移同时转换。此处是公共坐标基的转换，保持相对位姿、尺度与投影关系。VolSplat 在公共坐标中预测体素特征和高斯协方差，不能假定网络对参考系旋转不敏感。此项根据 2026-10-01 DDAD 低分排查补充，原方案直接沿用 DDAD 坐标的约定遗漏了这一适配。
4. SVF-GS 自身 `camera_tensors` 为其渲染器额外右乘 `diag(1,-1,-1,1)`；VolSplat 原 OmniScene 路径没有这一步。新加载器沿用 VolSplat 坐标约定，不能照搬该 OpenGL 翻转。
5. near/far 继续为 `0.5 / 100.0`。不根据新数据集深度分布重新设置或自动归一化尺度。

Batch 保持原契约，新增内容只由新包装器消费：

| 字段 | 单样本形状 / 用途 |
| --- | --- |
| `context.image` | `(6,3,H,W)` |
| `context.extrinsics / intrinsics` | `(6,4,4)` / `(6,3,3)` |
| `target.image` | `(18,3,H,W)` |
| `target.extrinsics / intrinsics` | `(18,4,4)` / `(18,3,3)` |
| 两侧 `near / far / index` | 分别长 6 / 18；index 按各自固定顺序编号 |
| `target.rel_depth` | `(18,H,W)`，Metric3D 尺度深度；仅需要 PCC 的评估路径加载 |
| `target.eval_mask` | `(18,H,W)` bool，仅 DDAD 评估；True 表示参与指标 |
| `scene` | bin token，兼容既有命名方式 |
| 新增样本位置元数据 | 原索引中的位置，用于结果记录及分布式采样补齐的统计处理 |

不提供动态 `target.masks`；`train.use_dynamic_mask=false`，所有训练像素都参与原损失。

### 3.5 patch shim 与真实评估尺寸

原 encoder 的 patch size 为 `shim_patch_size × downscale_factor = 4 × 4 = 16`。

| 加载尺寸 H×W | shim 后 H×W | 实际操作 |
| --- | --- | --- |
| 112×200 | 112×192 | 左右各中心裁去 4 列 |
| 224×400 | 224×400 | 不裁剪 |

保留原 `src/dataset/shims/patch_shim.py` 和 encoder 的 shim。新包装器组合一个仅用于 temporal18 的附加 shim，使 `target.eval_mask` 按同一 row/col 裁剪。原 shim 已处理 RGB、归一化内参和 `rel_depth`，继续使用它的结果；不重复裁剪，不将预测放大回 112×200。

掩码加载先在已处理的 224×400 空间最近邻缩放到 112×200，再裁成 112×192；不能直接将掩码拉伸到 112×192。结果同时保存 `load_resolution`、`evaluation_resolution`、`patch_crop`，避免将实际 112×192 结果表述为完整 112×200 评估。

## 4. 配置对齐

### 4.1 配置组织

新增数据集配置 `config/dataset/pandaset.yaml`、`ddad.yaml`，配置 dataclass 从 `DatasetCfgCommon` 扩展，只增加新数据集实际需要的根路径、near/far、评估深度和自车掩码字段。`view_sampler: all` 只用于满足框架接口；固定视角由加载器组织。

新增训练实验：

```text
config/experiment/pandaset_112x200.yaml
config/experiment/pandaset_224x400.yaml
config/experiment/ddad_112x200.yaml
config/experiment/ddad_224x400.yaml
```

通过 Hydra 组合继承相应的现有 OmniScene 实验，覆盖数据集、日志/输出名称、静态训练设置；模型使用 README 实际 Base 命令中的四项覆盖：`num_scales=2`、`upsample_factor=2`、`lowest_feature_resolution=4`、`monodepth_vit_type=vitb`。这些参数与最近的 OmniScene 测试快照一致，不能误用裸 YAML 中的 `vits / num_scales=1 / upsample_factor=4` 加载 Base 权重。

再新增四份 `*_zero_shot.yaml` 实验，继承相应新训练实验并设置 `mode=test`、权重入口、独立输出目录；保持训练字段完整，不另造简化模型或优化器配置。224×400 评估配置允许显式指定来源权重，并在输出中记录来源训练分辨率，不能把现有 112×200 权重描述为 224×400 训练权重。

### 4.2 模型参数基线

以下为“原 OmniScene 实验 + README Base 覆盖”的配置值；新数据集按同一组合继承，不借适配调整结构。

| 配置项 | 值 |
| --- | --- |
| encoder / decoder | `volsplat` / `splatting_cuda` |
| `monodepth_vit_type` | `vitb` |
| `num_scales / upsample_factor / lowest_feature_resolution` | `2 / 2 / 4` |
| `num_depth_candidates / num_surfaces / gaussians_per_pixel` | `128 / 1 / 1` |
| `d_feature / depth_unet_channels` | `128 / 128` |
| costvolume U-Net | feat dim 128；channel mult `[1,1,1]`；attn res `[4]` |
| depth U-Net 配置字段 | feat dim 32；channel mult `[1,1,1,1,1]`；attn res `[16]` |
| `downscale_factor / shim_patch_size` | `4 / 4` |
| `local_mv_match / multiview_trans_attn_split` | `2 / 2` |
| `voxel_resolution` | `0.001` |
| Gaussian adapter | scale min `1e-10`，scale max `3.0`，SH degree `2` |
| `supervise_intermediate_depth / return_depth` | `true / true` |
| `train_depth_only / grid_sample_disable_cudnn` | `false / false` |
| `large_gaussian_head / color_large_unet / init_sh_input_img` | `false / false / true` |
| `feature_upsampler_channels / gaussian_regressor_channels` 配置值 | `64 / 64`；不改变 encoder 内部由 ViT 类型确定实际通道的逻辑 |
| decoder 背景 | `[0,0,0]` |

其他 encoder/visualizer 字段完整继承原配置，不仅复制表内条目。配置字段是否参与当前 encoder 内部计算也保持原样，本次不清理或重解释旧参数。

### 4.3 训练及验证参数基线

| 项目 | 与 OmniScene 一致的设置 |
| --- | --- |
| 损失 | MSE weight `1.0`；LPIPS/VGG weight `0.05`，`apply_after_step=150000` |
| 中间输出 | 保持原监督路径，`intermediate_loss_weight=0.9` |
| 其他损失开关 | `l1_loss=false`、`train_ignore_large_loss=0`、`depth_mode=null`；不新增 Metric3D 深度损失 |
| 优化器 | AdamW；普通参数 `lr=2e-4`；名称含 `pretrained` 的参数组 `lr_monodepth=2e-6`；weight decay `0.01` |
| 调度器 | 原 `OneCycleLR`：total steps `max_steps+10`，`pct_start=0.01`，cosine，`cycle_momentum=false` |
| `warm_up_steps` | 配置继承 `2000`；当前 `configure_optimizers` 未用它构造调度器，不能据此另加 2000 步 warmup |
| batch size | train / val / test 均为 `1` |
| DataLoader workers | train `10`、val `1`、test `4`；persistent workers 与原配置一致 |
| 最大训练步数 | `100001` |
| 验证频率 | `val_check_interval=0.01`，表示每个训练 epoch 的 1%；不是固定 1000 步 |
| 训练期全量测试 | 每 10 次非 sanity 验证触发，`eval_data_length=999999`，`eval_deterministic=false` |
| 验证样本 | 原 `ValidationWrapper(...,1)`；sanity 验证配置 `num_sanity_val_steps=2` 原样保留 |
| checkpoint | 每 5000 步，`save_top_k=5`，按 `info/global_step` 的 max 规则；定期全量评估前保存备份 |
| 预训练初始化 | 保留原 monodepth/mvdepth 加载路径与冻结逻辑，以及 20000 步解冻回调 |
| 梯度裁剪 / 节点 | `0.5 / 1` |
| 种子 | root `111123`；loader train/test/val `1234/2345/3456`；保留原入口种子设置 |
| 计时跳过 | 训练期评估 `3`，独立测试 `5`；只影响计时统计，所有样本都计算图像指标 |
| 唯一必要的训练掩码差异 | PandaSet/DDAD `train.use_dynamic_mask=false` |

LPIPS 的启用步数晚于 100001 步训练上限，因此现有设置下它不会在此次完整训练计划内参与优化，但评估 LPIPS 正常计算。这里按用户要求保持与 OmniScene 一致，不擅自提前启用。

新数据集每个 epoch 的长度不同，所以“每 1% epoch 验证”的实际步间隔也不同。保留这个调度语义，不擅自改为按 nuScenes epoch 长度折算的固定步数。

## 5. DDAD 自车掩码及指标

### 5.1 掩码读取和使用范围

从 `processed/ego_masks/vidar_v1/manifest.json` 读取 `scene_camera_to_variant -> variants -> mask_id -> masks[path]`。掩码文件路径以 manifest 所在目录为基准。

按目标的 `scene_id` 和物理 `camera` 选择已有共享掩码；PNG 的 `255` 为有效、`0` 为排除区域。使用 PIL NEAREST 调整分辨率，并缓存已加载的 mask。直接使用既有派生掩码，不重新清理连通域、重算裁剪变换或读取原始模板。

`target.eval_mask[:12]` 来自对应新视角相机，`target.eval_mask[12:18]` 全 True。它只进入指标计算，不能写入 RGB、context、相机参数、Gaussian 预测或训练损失。LPIPS 的局部替换只操作指标函数内的临时预测张量。

用户已确认应用于所有评估，因此 DDAD 的 `validation_step`、训练期 `run_full_test_sets_eval`、独立 `test_step` 均使用同一计算函数。训练 DataLoader 不加载自车掩码，训练 loss 始终全图。

SVF-GS 当前的配置限制只允许独立 test 开启 ego mask；本项目按本次确认扩展到训练期评估，保持其掩码读取映射和指标数学定义，不照搬该模式限制。

### 5.2 与 SVF-GS 一致的指标定义

全图视角使用当前 PSNR、SSIM、VGG-LPIPS 原函数。即使提供了掩码，某个视角掩码全 True 时也回到原全图计算，保证 input_6 的指标口径不变。

部分掩码视角采用：

- **PSNR**：GT 和预测裁到 `[0,1]`；仅统计有效像素的三个通道平方误差。`MSE = sum(mask * squared_error) / (3 * sum(mask))`，再计算 `-10*log10(MSE)`，逐视角得到分数。
- **SSIM**：`win_size=11`、Gaussian weights、`sigma=1.5`、`use_sample_covariance=true`、`channel_axis=0`、`data_range=1.0`。取得完整 SSIM map；对有效掩码做 11×11 二值腐蚀，仅平均窗口全部落在有效区内的中心位置及三个通道。
- **LPIPS**：VGG、`spatial=true`、`normalize=true`。无效区域用 GT 替换临时预测中的对应像素，再计算空间 LPIPS map，仅对有效位置求均值；使用独立空间 LPIPS 实例，不更改原全图 LPIPS 实例。
- **PCC**：组内有效像素展平后计算 Pearson 相关系数；不平均逐视角 PCC 来替代组内展平计算。

不采用“把双方无效像素涂黑后做全图平均”的替代公式，不按有效像素面积改变视角间的权重。

这些公式的迁移不包括 SVF-GS 的数据审查异常。指标数学上未定义时保留该样本及该组，保存 `null` 和计算原因，不借此中断整次评估、剔除 bin 或把该分数写成 0；正无穷等值使用明确的可序列化表示。汇总不得静默删除未定义项后冒充完整均值。

### 5.3 分组及聚合

| 名称 | 目标索引 | DDAD 像素范围 |
| --- | --- | --- |
| `all_18` | `[0:18]` | 12 个新视角有效区 + 6 个输入视角全图 |
| `novel_12` | `[0:12]` | 各自自车掩码有效区 |
| `input_6` | `[12:18]` | 全图 |

PSNR/SSIM/LPIPS 先逐视角计算，再在每个 bin 内按组等权平均，最后对测试 bin 等权平均。不先混合所有像素再计算一个 PSNR。有限值情形下，三种 RGB 指标应满足 `all_18 = (12 * novel_12 + 6 * input_6) / 18`。

Metric3D 深度直接作为 `target.rel_depth` 的内容，按 RGB 对应的 resize/crop 对齐。它已经是尺度深度，不取倒数、不做 DA2 disparity 后处理或 min-max 归一化。decoder 使用 `depth_mode="depth"`；PCC 每个 bin、每个组单独展平计算，跨 bin 平均。报告元数据明确 `pcc_reference=metric3d_v2`，与原 OmniScene 的 DA2 PCC 来源区分。

### 5.4 输出约定

新数据集在独立输出目录内保存：

- `metrics/per_bin_metrics.csv`：每个测试索引位置三行，含 bin token、group、PSNR/SSIM/LPIPS、诊断 PCC、像素协议。
- `metrics/evaluation_summary.json`：三组均值、样本数、预期/实际处理数、运行是否完成，以及 checkpoint、来源数据集、配置、加载/评估尺寸、掩码协议、PCC 来源。
- 兼容本项目的 `scores_all_avg.json` 和 `scores_<metric>_all.json`：原平铺指标含义固定为 `all_18`；分组报告另行明确，不改变旧数据集文件。
- resolved config 和运行日志；训练期全量评估按 step 分目录，不覆盖独立测试结果。

DDAD 正式结果标注 `pixel_protocol=ddad_ego_novel12_v1`，目录使用 `_ego_novel12_v1` 后缀。PandaSet 为 `full_image`。PSNR 展示 3 位小数，SSIM/LPIPS 展示 4 位；JSON/CSV 保留计算精度。

正式测试默认不截断样本。分布式评估先汇集每个样本位置的记录，只消除 DistributedSampler 补齐产生的重复计数，不按 bin token 去重数据集。覆盖信息来自实际处理记录，不进行额外的数据集内容扫描；未完成运行保留已有记录并明确完成状态。

## 6. 实现文件及隔离方式

| 文件 | 计划改动 |
| --- | --- |
| `src/dataset/dataset_temporal18.py` | 新增共用的数据读取、train/val/test 索引与 batch 构造 |
| `src/dataset/dataset_pandaset.py`、`dataset_ddad.py` | 新增数据集名、配置 dataclass 与薄封装 |
| `src/dataset/temporal18/ego_mask.py` | 新增 DDAD 预处理掩码映射、读取及缓存，不包含审查逻辑 |
| `src/dataset/temporal18/shim.py` | 新增与原 patch crop 同步的评估掩码裁剪 |
| `src/evaluation/temporal18_metrics.py` | 新增局部掩码指标和三组聚合；复用原全图指标 |
| `src/evaluation/temporal18_reporting.py` | 新增逐样本记录、分布式汇集、结果写出 |
| `src/model/model_wrapper_temporal18.py` | 新增 `ModelWrapperTemporal18(ModelWrapper)`，仅新数据集使用 |
| `config/dataset/{pandaset,ddad}.yaml` | 新增数据集配置 |
| `config/experiment/{pandaset,ddad}_{112x200,224x400}.yaml` | 新增四份完整训练配置 |
| 对应 `*_zero_shot.yaml` | 新增四份评估配置，继承相应训练配置 |
| `src/dataset/__init__.py` | 仅添加两个新注册项和 DatasetCfg union 成员 |
| `src/main.py` | 仅新增 PandaSet/DDAD 固定视角 eval 配置分支与新 wrapper 选择；原数据集仍选择原实现 |
| `tests/test_temporal18_*.py` | 后续实现时新增必要的适配契约和数值测试，使用合成 fixture |
| README | 实施完成后追加新数据集命令，不改已有命令 |

专用 wrapper 继承原 `training_step`、`configure_optimizers`、训练验证计数及 checkpoint 节奏；在新类中实现带统一指标的 `validation_step / test_step / run_full_test_sets_eval / on_test_end`。继续调用原 encoder/decoder，并保留已有可视化开关的含义。新类不增加可训练模块或改变 state_dict 参数路径，保证 nuScenes 权重可以直接装载。

`src/model/model_wrapper.py`、`src/evaluation/metrics.py`、原 patch shim、原 DataModule、原数据集及其 utils、原 encoder/decoder/loss 和所有旧配置均不修改。入口的新增选择必须显式限定 `dataset.name in {"pandaset", "ddad"}`；旧分支的条件结果和执行顺序保持原样。不用 monkey patch，也不将旧数据集统一迁入新包装器。

## 7. 权重与命令设计

默认来源为已存在且被 README、最近测试配置共同引用的：

```text
checkpoints/omniscene-112x200-volsplat-base/checkpoints/epoch_0-step_100000.ckpt
```

零样本模式只加载完整模型参数，运行 eval/no-grad 推理，不恢复目标域优化过程，不调用 `fit`，不在目标数据集微调或更新 BatchNorm 统计。沿用现有 `checkpointing.pretrained_model` 入口；新 wrapper 对实际加载的模型参数匹配情况给出明确结果，避免错用裸 YAML 的 Small 配置后漏载 Base 权重。该检查针对模型配置和权重，不涉及数据集内容。

以下命令是待实现配置的预期接口，目前不执行：

```bash
# PandaSet，现有 nuScenes Base 权重，默认加载 112×200 / 评估 112×192
python -m src.main +experiment=pandaset_112x200_zero_shot \
  checkpointing.pretrained_model=checkpoints/omniscene-112x200-volsplat-base/checkpoints/epoch_0-step_100000.ckpt \
  output_dir=outputs/zero_shot/nuscenes_to_pandaset_112x200_base

# DDAD，相同权重；配置默认开启 novel_12 自车掩码
python -m src.main +experiment=ddad_112x200_zero_shot \
  checkpointing.pretrained_model=checkpoints/omniscene-112x200-volsplat-base/checkpoints/epoch_0-step_100000.ckpt \
  output_dir=outputs/zero_shot/nuscenes_to_ddad_112x200_base_ego_novel12_v1

# 后续若进行 PandaSet 训练，保持 OmniScene Base 初始化和训练设置
python -m src.main +experiment=pandaset_112x200 \
  checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vitb.pth \
  checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
  output_dir=checkpoints/pandaset-112x200-volsplat-base
```

DDAD 训练替换实验名和输出目录即可。224×400 使用对应配置；正式启动时显式指定来源权重和来源训练分辨率，不根据目标目录名推断或自动挑选另一模型。

## 8. 实施顺序与验收

1. **配置和注册**：新增四份训练、四份零样本配置及两个数据集注册项。用 Hydra 合成配置检查与对应 OmniScene Base 配置的差异，仅允许数据集、输出/日志、静态掩码及新评估字段不同。
2. **数据读取**：实现 train/val/test 索引、6/18 视角组织、归一化 K、直接 c2w、测试深度和 DDAD mask。所有读取按已有资产契约执行，不写预处理脚本。
3. **隔离评估**：新增 wrapper、掩码 shim、指标和输出模块；同时覆盖训练验证、训练期全量测试及独立测试，保持原训练方法及调度继承关系。
4. **必要的合成测试**：用小型合成资产验证索引规则、视角顺序、相机投影、112×200→112×192 同步裁剪、mask 只作用于 novel_12，以及三组均值关系。用固定 RGB/深度/mask 张量与 SVF-GS 数学实现逐项对照掩码 PSNR/SSIM/LPIPS/PCC；不进行真实数据内容审查。
5. **旧路径保持检查**：比较本轮起始工作区与适配后的受保护文件内容；对允许改动的注册表/入口检查旧数据集仍选原类、原配置和原分支。比较时以保留了用户未提交修改的工作区为基线，不以回退 HEAD 方式“恢复”旧文件。
6. **后续运行验证**：实施完成后，按获准的运行范围进行加载及推理验证，检查模型前向、权重加载和结果写出；不在本轮启动 GPU 实验。完整零样本结果需实际遍历 test，输出每个索引位置的三组记录。

实现不能新增“深度越界 / 非常量 / 哈希不同 / 内参不合预期 / manifest 不完整所以拒绝样本”等数据审查中断。遇到程序接口或实际文件读取错误按原始错误定位，不伪造数据、不静默跳过，也不自动修复或重生成数据资产。

本轮交付为本方案与已授权的两个软链接。关于权重来源、DDAD 掩码适用阶段和 patch 裁剪的决策均已确认；后续实施应遵守本文范围。

## 9. 本轮代码依据

VolSplat HEAD：`7bc56b954d0d53e7423f1fbfc7a913f7fc69133e`，另含上述用户已有的 OmniScene test 全量加载修改。

- [OmniScene 加载器](../src/dataset/dataset_omniscene.py)、[加载辅助函数](../src/dataset/utils_omniscene.py)
- [112×200 实验](../config/experiment/omniscene_112x200.yaml)、[224×400 实验](../config/experiment/omniscene_224x400.yaml)、[主配置](../config/main.yaml)
- [encoder 配置](../config/model/encoder/volsplat.yaml)、[LPIPS 配置](../config/loss/lpips.yaml)
- [入口](../src/main.py)、[DataModule](../src/dataset/data_module.py)、[ValidationWrapper](../src/dataset/validation_wrapper.py)
- [ModelWrapper](../src/model/model_wrapper.py)、[encoder](../src/model/encoder/encoder_volsplat.py)、[patch shim](../src/dataset/shims/patch_shim.py)、[指标](../src/evaluation/metrics.py)
- [本地 Base 训练/测试命令](../README.md)、最近测试配置 `outputs/2026-09-28/20-57-25/.hydra/config.yaml` 及 `overrides.yaml`

SVF-GS HEAD：`9a93216a11af6277dd3bd9c7d97abe89ff3e97fa`。以下均为本轮实际阅读的源代码，不依赖历史数据统计：

- `~/Projects/SVF-GS/data/temporal_dataset.py`：当前划分、18 视角组织和评估 mask 拼接。
- `~/Projects/SVF-GS/data/transforms/temporal_loading.py`：RGB/深度 resize 和相机约定。
- `~/Projects/SVF-GS/data/transforms/ego_mask.py`：DDAD 场景/相机到现成掩码的映射。
- `~/Projects/SVF-GS/configs/build_config.py`：相机映射及自车掩码指标参数。
- `~/Projects/SVF-GS/tools/metrics.py`、`tools/ablation_metrics.py`：掩码指标、组内/跨 bin 聚合。
- `~/Projects/SVF-GS/tools/temporal_data.py::make_bin_info`：预处理相机到中心 LiDAR 的坐标定义；仅读取代码，未运行预处理或校验。
