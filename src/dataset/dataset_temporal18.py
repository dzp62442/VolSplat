"""SVF-GS prepared PandaSet/DDAD assets, without preprocessing or data auditing."""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .dataset import DatasetCfgCommon
from .temporal18.ego_mask import DDADEgoMasks


CAMERAS = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")


@dataclass
class DatasetTemporal18Cfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    processed_root: Path | None = None
    near: float = 0.5
    far: float = 100.0
    compute_pcc: bool = True
    eval_use_ego_mask: bool = False
    ego_mask_manifest: str = "ego_masks/vidar_v1/manifest.json"


class DatasetTemporal18(Dataset):
    def __init__(self, cfg, stage, view_sampler):
        self.cfg, self.stage, self.view_sampler = cfg, stage, view_sampler
        self.root = cfg.processed_root or cfg.roots[0] / "processed"
        self.resolution = tuple(cfg.image_shape)
        self.load_depth = stage != "train" and cfg.compute_pcc
        storage_split = {"train": "train", "val": "test", "test": "test"}[stage]
        with (self.root / f"bins_{storage_split}.json").open() as handle:
            self.bin_tokens = json.load(handle)["bins"]
        if stage == "val":
            indices = np.linspace(0, len(self.bin_tokens) - 1, min(10, len(self.bin_tokens)), dtype=int)
            self.bin_tokens = [self.bin_tokens[i] for i in indices]
        self.ego_masks = None
        if cfg.name == "ddad" and stage != "train" and cfg.eval_use_ego_mask:
            self.ego_masks = DDADEgoMasks(self.root / cfg.ego_mask_manifest, self.resolution)

    def __len__(self):
        return len(self.bin_tokens)

    def _load_views(self, infos):
        images, intrinsics, depths = [], [], []
        height, width = self.resolution
        for info in infos:
            with Image.open(self.root / info["data_path"]) as source:
                source_w, source_h = source.size
                rgb = source.convert("RGB")
                if rgb.size != (width, height):
                    rgb = rgb.resize((width, height), Image.Resampling.BILINEAR)
                images.append(np.asarray(rgb, dtype=np.float32) / 255.0)
            with (self.root / info["intrinsic_path"]).open() as handle:
                k = np.asarray(json.load(handle)["camera_intrinsic"], dtype=np.float32)
            # Resize pixel K, then normalize; no OpenGL flip or baseline scaling.
            k[0] *= width / source_w
            k[1] *= height / source_h
            k[0] /= width
            k[1] /= height
            intrinsics.append(k)
            if self.load_depth:
                depth = np.load(self.root / info["depth_path"], allow_pickle=False)
                if depth.shape != (height, width):
                    depth = np.asarray(Image.fromarray(depth).resize((width, height), Image.Resampling.BILINEAR))
                depths.append(depth)
        count = len(infos)
        result = {
            "image": torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2),
            "intrinsics": torch.from_numpy(np.stack(intrinsics)),
            "extrinsics": torch.from_numpy(np.stack([np.asarray(i["sensor2lidar_transform"], dtype=np.float32) for i in infos])),
            "near": torch.full((count,), self.cfg.near),
            "far": torch.full((count,), self.cfg.far),
            "index": torch.arange(count, dtype=torch.int64),
        }
        if self.load_depth:
            result["rel_depth"] = torch.from_numpy(np.stack(depths).astype(np.float32))
        return result

    def __getitem__(self, index):
        token = self.bin_tokens[index]
        with (self.root / "bin_infos" / f"{token}.pkl").open("rb") as handle:
            info = pickle.load(handle)
        center = [info["sensor_info"][camera][0] for camera in CAMERAS]
        novel = [sensor for camera in CAMERAS for sensor in info["sensor_info"][camera][1:3]]
        context = self._load_views(center)
        target = self._load_views(novel)
        target = {key: torch.cat((value, context[key])) for key, value in target.items()}
        target["index"] = torch.arange(18, dtype=torch.int64)
        # The encoder receives RGB and cameras only, never the Metric3D reference.
        context.pop("rel_depth", None)
        if self.ego_masks is not None:
            target["eval_mask"] = torch.cat((self.ego_masks.load(info["scene_id"], novel),
                                              torch.ones((6, *self.resolution), dtype=torch.bool)))
        return {"context": context, "target": target, "scene": token,
                "sample_index": index, "scene_id": str(info["scene_id"])}

    def evaluation_metadata(self):
        return {"dataset": self.cfg.name, "split": self.stage, "expected_bins": len(self),
                "processed_root": str(self.root), "load_resolution": list(self.resolution),
                "protocol": "svfgs_temporal18_v1", "input_views": 6, "target_views": 18,
                "pcc_reference": "metric3d_v2" if self.load_depth else None,
                "pixel_protocol": "ddad_ego_novel12_v1" if self.ego_masks else "full_image"}
