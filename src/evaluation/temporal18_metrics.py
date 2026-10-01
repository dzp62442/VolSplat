"""SVF-GS image metric definitions for the new temporal18 datasets only."""

from functools import cache

import numpy as np
import torch
from lpips import LPIPS
from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity

from .metrics import compute_lpips, compute_pcc, compute_psnr, compute_ssim


VIEW_GROUPS = {"all_18": slice(0, 18), "novel_12": slice(0, 12), "input_6": slice(12, 18)}
MASK_METRIC_CONFIG = {"ssim_win_size": 11, "ssim_sigma": 1.5, "ssim_use_sample_covariance": True,
                      "lpips_net": "vgg", "lpips_normalize": True,
                      "lpips_rule": "gt_fill_invalid_spatial_valid_mean", "pcc_rule": "valid_group_flatten"}


@cache
def get_spatial_lpips(device):
    return LPIPS(net="vgg", spatial=True).to(device).eval()


@torch.no_grad()
def compute_image_metrics(ground_truth, predicted, mask=None):
    # Full views (including input_6) use exactly the existing metric functions.
    metrics = {name: fn(ground_truth, predicted) for name, fn in
               (("psnr", compute_psnr), ("ssim", compute_ssim), ("lpips", compute_lpips))}
    if mask is None:
        return metrics
    partial = ~mask.flatten(1).all(dim=1)
    if not partial.any():
        return metrics
    valid = mask[partial]
    gt, pred = ground_truth[partial], predicted[partial]
    squared = (gt.clip(0, 1) - pred.clip(0, 1)).square()
    mse = torch.where(valid[:, None], squared, 0).sum(dim=(1, 2, 3)) / (gt.shape[1] * valid.sum(dim=(1, 2)))
    metrics["psnr"][partial] = -10 * mse.log10()
    scores = []
    for target, prediction, area in zip(gt, pred, valid):
        centers = binary_erosion(area.cpu().numpy(), structure=np.ones((11, 11), dtype=bool))
        if not centers.any():
            scores.append(float("nan"))
            continue
        _, distance = structural_similarity(target.cpu().numpy(), prediction.cpu().numpy(),
                                           win_size=11, gaussian_weights=True, sigma=1.5,
                                           use_sample_covariance=True, channel_axis=0, data_range=1.0, full=True)
        scores.append(distance[:, centers].mean())
    metrics["ssim"][partial] = torch.as_tensor(scores, dtype=predicted.dtype, device=predicted.device)
    # This temporary tensor must never modify model inputs, outputs or loss tensors.
    pred_eval = torch.where(valid[:, None], pred, gt)
    distance = get_spatial_lpips(predicted.device)(gt, pred_eval, normalize=True)[:, 0]
    metrics["lpips"][partial] = torch.where(valid, distance, 0).sum(dim=(1, 2)) / valid.sum(dim=(1, 2))
    return metrics


@torch.no_grad()
def compute_group_pcc(ground_truth, predicted, mask=None):
    if mask is not None:
        ground_truth, predicted = ground_truth[mask], predicted[mask]
    if ground_truth.numel() < 2:
        return float("nan"), "fewer_than_two_metric_pixels"
    if ground_truth.var(unbiased=False) == 0 or predicted.var(unbiased=False) == 0:
        return float("nan"), "constant_depth_correlation_undefined"
    # Match SVF-GS's group-flattened torchmetrics computation, not per-view PCC.
    value = compute_pcc(ground_truth.reshape(1, 1, -1), predicted.reshape(1, 1, -1)).item()
    return value, None if np.isfinite(value) else "nonfinite_correlation"


@torch.no_grad()
def metric_records(batch, output, pixel_protocol, method="probabilistic"):
    records = []
    for b, token in enumerate(batch["scene"]):
        gt, pred = batch["target"]["image"][b], output.color[b]
        mask = batch["target"].get("eval_mask")
        mask = None if mask is None else mask[b]
        image_metrics = compute_image_metrics(gt, pred, mask)
        for group, indices in VIEW_GROUPS.items():
            row = {"sample_index": int(batch["sample_index"][b]), "bin_token": token,
                   "scene_id": batch["scene_id"][b], "view_group": group, "stage": "final",
                   "method": method, "pixel_protocol": pixel_protocol, "metric_issues": {}}
            for name, values in image_metrics.items():
                row[name] = values[indices].double().mean().item()
                if np.isnan(row[name]):
                    row["metric_issues"][name] = "undefined_view_score_or_empty_metric_support"
            row["pcc"] = None
            if "rel_depth" in batch["target"] and output.depth is not None:
                row["pcc"], issue = compute_group_pcc(batch["target"]["rel_depth"][b, indices],
                                                     output.depth[b, indices], None if mask is None else mask[indices])
                if issue:
                    row["metric_issues"]["pcc"] = issue
            records.append(row)
    return records
