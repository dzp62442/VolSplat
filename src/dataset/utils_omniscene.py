import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image


def _hwc3(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = image[:, :, None]
    if image.shape[2] == 3:
        return image
    if image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    if image.shape[2] == 4:
        color = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        merged = color * alpha + 255.0 * (1.0 - alpha)
        return merged.clip(0, 255).astype(np.uint8)
    raise ValueError(f"Unsupported channel count: {image.shape}")


def load_info(info: dict) -> Tuple[str, np.ndarray, np.ndarray]:
    img_path = info["data_path"]
    c2w = info["sensor2lidar_transform"]

    lidar2cam_r = np.linalg.inv(info["sensor2lidar_rotation"])
    lidar2cam_t = info["sensor2lidar_translation"] @ lidar2cam_r.T
    w2c = np.eye(4)
    w2c[:3, :3] = lidar2cam_r.T
    w2c[3, :3] = -lidar2cam_t

    return img_path, c2w, w2c


def load_conditions(
    img_paths: list[str],
    resolution: Tuple[int, int],
    is_input: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def _maybe_resize(image: Image.Image, ck: np.ndarray):
        resize_flag = False
        if image.height != resolution[0] or image.width != resolution[1]:
            fx, fy, cx, cy = ck[0, 0], ck[1, 1], ck[0, 2], ck[1, 2]
            scale_h, scale_w = resolution[0] / image.height, resolution[1] / image.width
            fx_scaled, fy_scaled = fx * scale_w, fy * scale_h
            cx_scaled, cy_scaled = cx * scale_w, cy * scale_h
            ck = np.array([[fx_scaled, 0, cx_scaled], [0, fy_scaled, cy_scaled], [0, 0, 1]])
            image = image.resize((resolution[1], resolution[0]))
            resize_flag = True
        return np.array(image), ck, resize_flag

    images, intrinsics, masks = [], [], []
    for img_path in img_paths:
        param_path = (
            img_path.replace("samples", "samples_param_small")
            .replace("sweeps", "sweeps_param_small")
            .replace(".jpg", ".json")
        )
        ck = np.array(json.load(open(param_path))["camera_intrinsic"])

        actual_img_path = (
            img_path.replace("samples", "samples_small")
            .replace("sweeps", "sweeps_small")
        )
        image = Image.open(actual_img_path)
        image_np, ck, resized = _maybe_resize(image, ck)
        ck[0, :] = ck[0, :] / resolution[1]
        ck[1, :] = ck[1, :] / resolution[0]
        images.append(_hwc3(image_np))
        intrinsics.append(ck)

        if is_input:
            mask = np.ones(resolution, dtype=np.float32)
        else:
            mask_path = (
                actual_img_path.replace("sweeps_small", "sweeps_mask_small")
                .replace("samples_small", "samples_mask_small")
                .replace(".jpg", ".png")
            )
            mask_image = Image.open(mask_path).convert("L")
            if resized:
                mask_image = mask_image.resize((resolution[1], resolution[0]), Image.BILINEAR)
            mask = np.array(mask_image).astype(np.float32) / 255.0
        masks.append(mask)

    images = torch.from_numpy(np.stack(images, axis=0)).permute(0, 3, 1, 2).float() / 255.0
    intrinsics = torch.as_tensor(np.stack(intrinsics, axis=0), dtype=torch.float32)
    masks = torch.from_numpy(np.stack(masks, axis=0)).bool()
    return images, masks, intrinsics
