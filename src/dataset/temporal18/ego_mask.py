"""Read prepared DDAD mask mappings; trust SVF-GS preprocessing."""

import json

import numpy as np
import torch
from PIL import Image


class DDADEgoMasks:
    def __init__(self, manifest_path, resolution):
        self.root = manifest_path.parent
        with manifest_path.open() as handle:
            self.manifest = json.load(handle)
        self.resolution = resolution
        self.cache = {}

    def load(self, scene, infos):
        masks = []
        for info in infos:
            variant_key = self.manifest["scene_camera_to_variant"][scene][info["camera"]]
            mask_id = self.manifest["variants"][variant_key]["mask_id"]
            if mask_id not in self.cache:
                path = self.root / self.manifest["masks"][mask_id]["path"]
                with Image.open(path) as source:
                    resized = source.resize(self.resolution[::-1], Image.Resampling.NEAREST)
                    self.cache[mask_id] = torch.from_numpy(np.asarray(resized) == 255)
            masks.append(self.cache[mask_id])
        return torch.stack(masks)
