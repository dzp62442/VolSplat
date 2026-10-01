def with_eval_mask_crop(base_shim):
    """Keep the original image/depth/K shim and crop only the new eval mask."""
    def shim(batch):
        mask = batch["target"].get("eval_mask")
        result = base_shim(batch)
        if mask is not None:
            h, w = result["target"]["image"].shape[-2:]
            row, col = (mask.shape[-2] - h) // 2, (mask.shape[-1] - w) // 2
            result["target"]["eval_mask"] = mask[..., row:row + h, col:col + w]
        return result
    return shim
