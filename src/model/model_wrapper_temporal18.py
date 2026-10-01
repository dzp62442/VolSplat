"""PandaSet/DDAD evaluation; original training, optimizer and scheduling are inherited."""

from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from pytorch_lightning.utilities import rank_zero_only

from .model_wrapper import ModelWrapper
from .decoder.decoder import DecoderOutput
from ..dataset.temporal18.shim import with_eval_mask_crop
from ..evaluation.temporal18_metrics import MASK_METRIC_CONFIG, metric_records
from ..evaluation.temporal18_reporting import EvaluationWriter, write_json
from ..global_cfg import get_cfg
from ..misc.image_io import prep_image, save_image, save_video
from ..visualization.annotation import add_label
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_cameras, render_projections
from ..visualization.vis_depth import viz_depth_tensor


class ModelWrapperTemporal18(ModelWrapper):
    def __init__(self, *args, dataset_cfg, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_cfg = dataset_cfg
        # The original typed loader does not resolve OmegaConf interpolations.
        self.test_cfg.output_path = Path(get_cfg().test.output_path)
        self.data_shim = with_eval_mask_crop(self.data_shim)
        self._weights_loaded = False
        self._test_writer = None

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # The existing entrypoint loads full pretrained models with strict=False.
        # For zero-shot evaluation, never silently leave model parameters random.
        required = {name for name, _ in self.named_parameters() if name.startswith(("encoder.", "decoder."))}
        missing = required - state_dict.keys()
        unexpected = {name for name in state_dict if name.startswith(("encoder.", "decoder."))} - self.state_dict().keys()
        if missing or unexpected:
            raise RuntimeError(f"Temporal18 checkpoint/model mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._weights_loaded = True
        print(f"Temporal18: loaded all {len(required)} encoder/decoder parameter tensors.")
        return result

    def setup(self, stage):
        if self.global_rank == 0:
            directory = Path(get_cfg()["output_dir"])
            directory.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(get_cfg(), directory / "resolved_config.yaml")

    @property
    def pixel_protocol(self):
        return "ddad_ego_novel12_v1" if self.dataset_cfg.name == "ddad" and self.dataset_cfg.eval_use_ego_mask else "full_image"

    def _metadata(self, dataset):
        cfg = get_cfg()
        h, w = dataset.resolution
        patch = self.encoder.cfg.shim_patch_size * self.encoder.cfg.downscale_factor
        height, width = h // patch * patch, w // patch * patch
        return {**dataset.evaluation_metadata(), "evaluation_resolution": [height, width],
                "patch_crop": {"top": (h - height) // 2, "left": (w - width) // 2, "height": height, "width": width},
                "checkpoint": cfg.checkpointing.load or cfg.checkpointing.pretrained_model,
                "source_checkpoint": OmegaConf.to_container(cfg, resolve=True).get("source_checkpoint"),
                "evaluation_step": self.global_step, "mask_metrics": MASK_METRIC_CONFIG if self.pixel_protocol != "full_image" else None,
                "resolved_config": OmegaConf.to_container(cfg, resolve=True)}

    @torch.no_grad()
    def _render(self, batch, deterministic=False):
        with self.benchmarker.time("encoder"):
            encoded = self.encoder(batch["context"], self.global_step, deterministic=deterministic, scene_names=batch["scene"])
        depths = encoded.get("depths") if isinstance(encoded, dict) else None
        gaussians = encoded["gaussians"] if isinstance(encoded, dict) else encoded
        if self.train_cfg.forward_depth_only:
            return None, gaussians, depths
        target = batch["target"]
        views = target["image"].shape[1]
        chunk = self.test_cfg.render_chunk_size or views
        outputs = []
        with self.benchmarker.time("decoder", num_calls=views):
            for start in range(0, views, chunk):
                selected = slice(start, start + chunk)
                outputs.append(self.decoder.forward(gaussians, target["extrinsics"][:, selected],
                               target["intrinsics"][:, selected], target["near"][:, selected], target["far"][:, selected],
                               tuple(target["image"].shape[-2:]), depth_mode="depth" if "rel_depth" in target else None))
        output = DecoderOutput(torch.cat([part.color for part in outputs], dim=1),
                               None if outputs[0].depth is None else torch.cat([part.depth for part in outputs], dim=1))
        return output, gaussians, depths

    def on_test_start(self):
        if not self._weights_loaded:
            raise RuntimeError("Temporal18 testing requires a complete pretrained checkpoint.")
        if self.test_cfg.stablize_camera:
            raise ValueError("Temporal18 evaluates the prepared target poses; stablize_camera must remain false.")
        loader = self.trainer.test_dataloaders
        if isinstance(loader, (list, tuple)):
            loader = loader[0]
        self._test_writer = EvaluationWriter(self.test_cfg.output_path, self._metadata(loader.dataset), self.global_rank)
        self.benchmarker.clear_history()

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        batch = self.data_shim(batch)
        output, gaussians, depths = self._render(batch)
        if output is not None and self.test_cfg.compute_scores:
            rows = metric_records(batch, output, self.pixel_protocol)
            self._test_writer.append(rows)
            for row in rows:
                print(f"{row['bin_token']} {row['view_group']}: PSNR={row['psnr']:.3f} "
                      f"SSIM={row['ssim']:.4f} LPIPS={row['lpips']:.4f}")
        self._save_test_visuals(batch, output, gaussians, depths)

    def on_test_end(self):
        records = self._test_writer.records
        if dist.is_available() and dist.is_initialized():
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, records)
            records = [row for rank_records in gathered for row in rank_records]
        if self.global_rank == 0:
            summary = self._test_writer.finish(records)
            print("Temporal18 summary:", summary["groups"])
            self._save_timing(Path(self.test_cfg.output_path), self.test_cfg.eval_time_skip_steps)
        self._test_writer = None
        self.benchmarker.clear_history()

    def _save_timing(self, directory, skip):
        timings = {}
        for tag, values in self.benchmarker.execution_times.items():
            values = values[skip * (18 if tag == "decoder" else 1):]
            timings[tag] = {"calls": len(values), "mean_seconds": float(np.mean(values)) if values else None}
        write_json(directory / "timing.json", {"boundary": "unsynchronized host encoder/decoder calls; excludes metrics",
                   "rank": self.global_rank, "skip_bins": skip, "timings": timings})

    @rank_zero_only
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        batch = self.data_shim(batch)
        output, gaussians, depths = self._render(batch)
        if output is not None:
            rows = metric_records(batch, output, self.pixel_protocol)
            self._log_groups(rows, "val")
            self._validation_visuals(batch, output, gaussians, depths)

    def _log_groups(self, rows, prefix):
        for row in rows:
            for name in ("psnr", "ssim", "lpips", "pcc"):
                value = row[name]
                if value is not None:
                    self.log(f"{prefix}/{row['view_group']}/{name}", value, batch_size=1, rank_zero_only=True)
                    if row["view_group"] == "all_18":
                        key = f"val/{name}_val" if prefix == "val" else f"test/{name}"
                        self.log(key, value, batch_size=1, rank_zero_only=True)

    @rank_zero_only
    @torch.no_grad()
    def run_full_test_sets_eval(self):
        loader = self.trainer.datamodule.test_dataloader()
        directory = Path(get_cfg()["output_dir"]) / "evaluation" / f"step-{self.global_step:08d}"
        writer = EvaluationWriter(directory, self._metadata(loader.dataset))
        was_training = self.training
        self.eval()
        self.benchmarker.clear_history()
        started = time.time()
        try:
            for index, batch in enumerate(loader):
                if index >= self.train_cfg.eval_data_length:
                    break
                batch = self.data_shim(self.transfer_batch_to_device(batch, self.device, 0))
                output, _, _ = self._render(batch)
                if output is not None:
                    writer.append(metric_records(batch, output, self.pixel_protocol))
                if self.train_cfg.eval_deterministic:
                    output, _, _ = self._render(batch, deterministic=True)
                    if output is not None:
                        writer.append(metric_records(batch, output, self.pixel_protocol, method="deterministic"))
            summary = writer.finish()
            for group, values in summary["groups"].items():
                self._log_groups([{"view_group": group, **{key: values[key] for key in ("psnr", "ssim", "lpips", "pcc")}}], "test")
            self._save_timing(directory, self.train_cfg.eval_time_skip_steps)
            self.log("test/runtime_all", time.time() - started, rank_zero_only=True)
        finally:
            self.train(was_training)
            self.benchmarker.clear_history()

    def _validation_visuals(self, batch, output, gaussians, depths):
        if self.logger is None:
            return
        if depths is not None:
            depth = depths[0]
            if depth.shape[-2:] != batch["context"]["image"].shape[-2:]:
                depth = F.interpolate(depth[:, None], size=batch["context"]["image"].shape[-2:], mode="bilinear", align_corners=True)[:, 0]
            depth_image = viz_depth_tensor(torch.cat(list(1.0 / depth), dim=1).cpu())
            rgb = torch.cat(list(batch["context"]["image"][0]), dim=-1).cpu() * 255
            self.logger.log_image("depth", [torch.cat((rgb, depth_image), dim=1)], step=self.global_step, caption=batch["scene"])
        comparison = hcat(add_label(vcat(*batch["context"]["image"][0]), "Context"),
                          add_label(vcat(*batch["target"]["image"][0]), "Target (Ground Truth)"),
                          add_label(vcat(*output.color[0]), "Target (Prediction)"))
        self.logger.log_image("comparison", [prep_image(add_border(comparison))], step=self.global_step, caption=batch["scene"])
        if not self.train_cfg.no_log_projections:
            projections = hcat(*render_projections(gaussians, 256, extra_label="(Prediction)")[0])
            self.logger.log_image("projection", [prep_image(add_border(projections))], step=self.global_step)
            self.logger.log_image("cameras", [prep_image(hcat(*render_cameras(batch, 256)))], step=self.global_step)
        if self.encoder_visualizer is not None:
            for key, image in self.encoder_visualizer.visualize(batch["context"], self.global_step).items():
                self.logger.log_image(key, [prep_image(image)], step=self.global_step)
        if not self.train_cfg.no_viz_video:
            self.render_video_interpolation(batch)
            self.render_video_wobble(batch)
            if self.train_cfg.extended_visualization:
                self.render_video_interpolation_exaggerated(batch)

    def _save_test_visuals(self, batch, output, gaussians, depths):
        root = Path(get_cfg()["output_dir"])
        for b, token in enumerate(batch["scene"]):
            directory = root / "images" / token
            if self.test_cfg.save_input_images:
                for index, image in enumerate(batch["context"]["image"][b]):
                    save_image(image, directory / "color" / f"input_{index:06d}.png")
            if output is not None:
                for index, (prediction, gt) in enumerate(zip(output.color[b], batch["target"]["image"][b])):
                    if self.test_cfg.save_image:
                        save_image(prediction, directory / "color" / f"{index:06d}.png")
                    if self.test_cfg.save_gt_image:
                        save_image(gt, directory / "color" / f"{index:06d}_gt.png")
                if self.test_cfg.save_video:
                    save_video(list(output.color[b]), root / "videos" / f"{token}.mp4")
            if depths is not None and (self.test_cfg.save_depth or self.test_cfg.save_depth_npy or self.test_cfg.save_depth_concat_img):
                depth = depths[b]
                if depth.shape[-2:] != batch["context"]["image"].shape[-2:]:
                    depth = F.interpolate(depth[:, None], size=batch["context"]["image"].shape[-2:], mode="bilinear", align_corners=True)[:, 0]
                (directory / "depth").mkdir(parents=True, exist_ok=True)
                colors = []
                for index, value in enumerate(depth):
                    color = viz_depth_tensor((1.0 / value).cpu(), return_numpy=True)
                    colors.append(color)
                    if self.test_cfg.save_depth:
                        from PIL import Image
                        Image.fromarray(color).save(directory / "depth" / f"{index:06d}.png")
                    if self.test_cfg.save_depth_npy:
                        np.save(directory / "depth" / f"{index:06d}.npy", value.cpu().numpy())
                if self.test_cfg.save_depth_concat_img:
                    from PIL import Image
                    rgb = prep_image(torch.cat(list(batch["context"]["image"][b]), dim=-1))
                    Image.fromarray(np.concatenate((rgb, np.concatenate(colors, axis=1)), axis=0)).save(directory / "depth" / "input_depth.png")
            if self.test_cfg.save_gaussian:
                from ..evaluation.temporal18_visualization import save_gaussians
                save_gaussians(gaussians, b, root / "gaussians" / f"{token}.ply")
