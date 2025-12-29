import json
import os
import os.path as osp
import pickle as pkl
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset

from .dataset import DatasetCfgCommon
from .types import Stage
from .view_sampler import ViewSampler
from .utils_omniscene import load_conditions, load_info


@dataclass
class DatasetOmniSceneCfg(DatasetCfgCommon):
    name: Literal["omniscene"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    train_times_per_scene: int
    highres: bool
    skip_bad_shape: bool = True
    near: float = 0.5
    far: float = 100.0
    baseline_scale_bounds: bool = False
    shuffle_val: bool = True
    test_chunk_interval: int = 1
    use_index_to_load_chunk: bool = False


class DatasetOmniScene(Dataset):
    cfg: DatasetOmniSceneCfg
    stage: Stage
    view_sampler: ViewSampler

    data_version: str = "interp_12Hz_trainval"
    dataset_prefix: str = "/datasets/nuScenes"
    camera_types = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    def __init__(
        self,
        cfg: DatasetOmniSceneCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.resolution = tuple(cfg.image_shape)
        self.data_root = str(cfg.roots[0])
        self.near = cfg.near
        self.far = cfg.far
        self.bin_tokens = self._load_bin_tokens()

    def _load_bin_tokens(self) -> list[str]:
        meta_root = osp.join(self.data_root, self.data_version)
        if self.stage == "train":
            file_path = osp.join(meta_root, "bins_train_3.2m.json")
            return json.load(open(file_path))["bins"]
        if self.stage == "val":
            file_path = osp.join(meta_root, "bins_val_3.2m.json")
            subset = json.load(open(file_path))["bins"]
            return subset[:30000:3000][:10]
        if self.stage == "test":
            file_path = osp.join(meta_root, "bins_val_3.2m.json")
            subset = json.load(open(file_path))["bins"]
            return subset[0::14][:2048]
        raise ValueError(f"Unsupported stage: {self.stage}")

    def __len__(self) -> int:
        return len(self.bin_tokens)

    def __getitem__(self, index: int):
        bin_token = self.bin_tokens[index]
        info_path = osp.join(
            self.data_root,
            self.data_version,
            "bin_infos_3.2m",
            bin_token + ".pkl",
        )
        with open(info_path, "rb") as f:
            bin_info = pkl.load(f)

        sensor_info_center = {
            sensor: bin_info["sensor_info"][sensor][0]
            for sensor in self.camera_types + ["LIDAR_TOP"]
        }

        input_img_paths, input_c2ws = [], []
        for cam in self.camera_types:
            info = sensor_info_center[cam]
            img_path, c2w, _ = load_info(info)
            img_path = img_path.replace(self.dataset_prefix, self.data_root)
            input_img_paths.append(img_path)
            input_c2ws.append(c2w)
        input_c2ws = torch.as_tensor(input_c2ws, dtype=torch.float32)
        input_imgs, input_masks, input_cks = load_conditions(
            input_img_paths, self.resolution, is_input=True
        )
        input_cks = torch.as_tensor(input_cks, dtype=torch.float32)

        frame_num = len(bin_info["sensor_info"]["LIDAR_TOP"])
        if frame_num < 3:
            raise ValueError(f"bin {bin_token} has only {frame_num} frames")
        rend_indices = [[1, 2]] * len(self.camera_types)

        output_img_paths, output_c2ws = [], []
        for cam_id, cam in enumerate(self.camera_types):
            indices = rend_indices[cam_id]
            for ind in indices:
                info = bin_info["sensor_info"][cam][ind]
                img_path, c2w, _ = load_info(info)
                img_path = img_path.replace(self.dataset_prefix, self.data_root)
                output_img_paths.append(img_path)
                output_c2ws.append(c2w)
        output_c2ws = torch.as_tensor(output_c2ws, dtype=torch.float32)
        output_imgs, output_masks, output_cks = load_conditions(
            output_img_paths, self.resolution, is_input=False
        )
        output_cks = torch.as_tensor(output_cks, dtype=torch.float32)

        output_imgs = torch.cat([output_imgs, input_imgs], dim=0)
        output_masks = torch.cat([output_masks, input_masks], dim=0)
        output_c2ws = torch.cat([output_c2ws, input_c2ws], dim=0)
        output_cks = torch.cat([output_cks, input_cks], dim=0)

        context = {
            "extrinsics": input_c2ws,
            "intrinsics": input_cks,
            "image": input_imgs,
            "near": torch.full((len(self.camera_types),), self.near, dtype=torch.float32),
            "far": torch.full((len(self.camera_types),), self.far, dtype=torch.float32),
            "index": torch.arange(len(self.camera_types), dtype=torch.int64),
        }

        target = {
            "extrinsics": output_c2ws,
            "intrinsics": output_cks,
            "image": output_imgs,
            "near": torch.full((output_c2ws.shape[0],), self.near, dtype=torch.float32),
            "far": torch.full((output_c2ws.shape[0],), self.far, dtype=torch.float32),
            "index": torch.arange(output_c2ws.shape[0], dtype=torch.int64),
            "masks": output_masks,
        }

        return {
            "context": context,
            "target": target,
            "scene": bin_token,
        }
