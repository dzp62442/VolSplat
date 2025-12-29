# OmniScene 数据集实验规划

## 配置概览
- **数据集配置文件**：新增 `config/dataset/omniscene.yaml`，结构沿用 `re10k`，`name` 设为 `omniscene`、`roots` 指向 `datasets/omniscene`，`defaults.view_sampler=all`，以便直接使用我们在 DepthSplat 中约定好的固定视角顺序。图像默认分辨率填写 `image_shape: [224, 400]`，背景色设为黑色（`[0, 0, 0]`），`near/far` 保持 0.5 / 100.0，与 DepthSplat 中的训练设置一致，`baseline_scale_bounds=false` 防止自动缩放 near/far。
- **实验配置文件**：参考 `config/experiment/re10k.yaml` 的写法复制两份：`config/experiment/omniscene_112x200.yaml` 与 `config/experiment/omniscene_224x400.yaml`。两者的共同点：
  - `defaults` 里覆盖 `dataset: omniscene`、`model/encoder: volsplat`、`loss: [mse, lpips]`，保留 VolSplat 原有的编码器/解码器结构与损失组合。
  - `data_loader.train.batch_size=1`，`data_loader.val.batch_size` 与 `data_loader.test.batch_size` 已在 `config/main.yaml` 中默认为 1，可不额外指定。OmniScene 的场景分辨率高、单个样本包含 18 张输出视图，因此训练/验证/测试都固定 batch size 为 1。
  - `trainer.max_steps=100_001` 与 DepthSplat 保持一致；`trainer.val_check_interval=0.01` 表示每 1% epoch 触发一次验证。
  - `train.eval_model_every_n_val=10`：沿用 DepthSplat 的测试节奏；因为 VolSplat 的 `src/main.py` 会使用该字段在训练过程中触发 `trainer.test`，因此直接在实验配置覆盖即可。若后续我们不需要训练期测试，可将该字段设置为 0。
  - `loss.lpips.weight=0.05`，与 re10k 实验一致，保证训练指标对齐。
  - `trainer.num_nodes=1`，`wandb.name/tags` 区分分辨率（112x200 或 224x400）。
- **分辨率差异**：
  - `omniscene_112x200`：在实验配置里额外设置 `dataset.image_shape=[112,200]`，保持 near/far 以及 view_sampler 与 224x400 版本一致。
  - `omniscene_224x400`：保留数据集配置默认值（224x400）。
- **训练/测试指令**：README 中 re10k 的命令均以 `python -m src.main +experiment=re10k ...` 形式运行。我们将在文档/实验脚本中提供对应命令，例如：
  ```bash
  python -m src.main +experiment=omniscene_224x400 \
    data_loader.train.batch_size=1 \
    trainer.max_steps=100001 \
    trainer.val_check_interval=0.01 \
    train.eval_model_every_n_val=10 \
    model.encoder.num_scales=2 \
    model.encoder.upsample_factor=2 \
    model.encoder.lowest_feature_resolution=4 \
    model.encoder.monodepth_vit_type=vitb \
    checkpointing.pretrained_monodepth=pretrained/pretrained_weights/depth_anything_v2_vitb.pth \
    checkpointing.pretrained_mvdepth=pretrained/pretrained_weights/gmflow-scale1-things-e9887eda.pth \
    output_dir=outputs/omniscene-224x400
  ```
  112x200 版本仅需改 `+experiment`、`dataset.image_shape`，其余模型超参、学习率、损失等保持不变，确保和 re10k 的配置兼容。

## 数据加载方案
1. **类结构与入口**：
   - 仿照 DepthSplat，将 `DatasetOmniScene` 及其配置定义放在 `src/dataset/dataset_omniscene.py`，并在 `src/dataset/__init__.py` 的 `DATASETS` 字典里注册 `"omniscene": DatasetOmniScene`。配置 dataclass 复用 `DatasetCfgCommon`，新增 `test_len`、`highres` 等字段以便与 re10k 接口对齐。
   - 虽然 VolSplat 的 re10k/dl 数据集依赖 `ViewSampler`，OmniScene 的输入/输出视角完全由数据本身决定（6 个 key-frame 输入 + 每个视角 2 帧输出）。因此 `DatasetOmniScene` 内部不需要 `view_sampler.sample`，只需保留参数占位即可；配置中将 `defaults.view_sampler=all`，确保 Hydra 仍能解析到一个合法的 view sampler 实例。
2. **Bin 管理与划分**：
   - 沿用 DepthSplat 的约定：`train` 读取 `bins_train_3.2m.json` 全量 bin，`val` 选取 `bins_val_3.2m.json` 的稀疏子集（前 30000 个里每隔 3000 取一个，再取 10 个），`test` 默认 mini-test（`0::14` 取 2048 个）。若需要完整测试集，可在配置或命令参数中新增开关；实现阶段会把该逻辑写成可配置项。
3. **图像、相机与掩码加载**：
   - `load_info`、`load_conditions`、`get_rays` 等辅助函数可以直接从 DepthSplat 复制，只需确认路径前缀改为 VolSplat 约定的根目录。由于两边都采用单位化的内参表示（fx、fy 分别除以宽/高），现有实现可复用。
   - `load_conditions` 负责读取 JPEG、缩放/归一化内参，并在输出视图加载动态掩码。输入视图用全 1 掩码即可。
   - `__getitem__`：固定 6 个摄像头视角作为 `context`，输出由每个摄像头的第 1、2 帧以及输入帧拼接组成，总共 18 张。所有字段（`extrinsics`、`intrinsics`、`near/far`、`index`、`image`、`masks`）与 DepthSplat 返回结构保持一致，模型即可直接消费。
4. **与 DepthSplat 的差异**：
   - VolSplat 的 re10k 数据集本质上是 `IterableDataset`，依赖 chunk 与 view sampler；OmniScene 的实现会更像 DepthSplat：继承普通 `Dataset`、一次返回完整的上下文/监督视图。`DataModule` 已经能处理这两种情况，所以可以共存。
   - 现有 `src/dataset/shims/patch_shim.py` 只裁剪图像和内参，没有处理掩码。为保持与 DepthSplat 一致，我们需要扩展该 shim：当视图包含 `masks` 字段时同步裁剪掩码，确保动态区域与裁剪后的图像对齐。
   - `target` 中的掩码体积可能较大，会成为 dataloader 的额外内存占用。为避免重复拷贝，需要确保 `DatasetOmniScene` 返回 `torch.bool` 类型，并尽量复用 Tensor 存储（与 DepthSplat 相同）。

## 主程序接入与差异
1. **DataModule / Trainer**：VolSplat 的 `src/main.py` 在 `train.eval_model_every_n_val>0` 时，会复制一份配置赋给 `eval_cfg`，并以当前数据集根目录判断 eval index。OmniScene 没有额外的 evaluation index json，因此我们计划在配置中显式写入 `dataset.view_sampler`（例如 `all`），同时在 `src/main.py` 中检测到 `dataset.name=omniscene` 时跳过自动注入 eval index，直接使用训练集同样的采样方式即可。
2. **动态掩码**：
   - `ModelWrapper.training_step` 已经支持 `valid_depth_mask` 参数并在 `LossMse/LossLpips` 内判断，但当前实现始终传入 `None`。我们会参照 DepthSplat 的写法，给 `TrainCfg` 新增 `use_dynamic_mask: bool`，在 `config/main.yaml` 里默认为 `false`，在 OmniScene 的实验配置中置为 `true`。
   - 当该开关生效时，从 `batch["target"]["masks"]` 生成布尔掩码，反转后作为 `valid_depth_mask` 注入各个损失（黑色区域即动态区域被排除在监督之外）。这样既能与 DepthSplat 保持一致，也不会影响其它数据集的训练流程。
3. **命令行调用**：
   - 训练与测试命令沿用 README 中 re10k 的格式，只需把 `+experiment` 换成 `omniscene_112x200/224x400`，并按需覆写 `model.encoder` 的尺度参数（small/base/large）。
   - Checkpoint/日志目录建议也命名为 `outputs/omniscene-<reso>-volsplat-*`，方便与 DepthSplat 的实验对齐。
4. **无法直接复用的部分**：
   - DepthSplat 的 `DatasetOmniScene` 中包含 `get_rays`、`load_rel_depth` 以及自定义的 `test.save_video_omniscene` 可视化，VolSplat 暂时不需要在第一阶段实现这些功能；若后续需要渲染环视视频，可在完成基础训练后再补充。
   - DepthSplat 默认在 `README` 中通过命令行直接设置 `train.use_dynamic_mask=true`；VolSplat 需要在配置层面显式新增该字段以保证 Hydra 校验通过。

## 小结
OmniScene 的整体实现可以基于 DepthSplat 的成熟代码迁移到 VolSplat：配置层面保持与 re10k 相同的模型/优化器设置，仅针对 batch size、分辨率与训练节奏做覆盖；数据层面复刻 DepthSplat 的加载逻辑并补全 mask 的 shim；主程序层面增加可选的动态掩码开关和 OmniScene 专用的 eval 行为。待本规划通过审阅后，再按本文档的步骤依次提交配置、数据集类与主程序改动，即可在 VolSplat 的 `comp_svfgs` 分支开启 OmniScene 实验，与自研方法进行公平对比。
