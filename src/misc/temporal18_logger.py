"""Local image logger scoped to a new dataset's configured output directory."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities import rank_zero_only


class Temporal18LocalLogger(Logger):
    def __init__(self, output_dir):
        super().__init__()
        self.directory = Path(output_dir) / "local"

    @property
    def name(self):
        return "Temporal18LocalLogger"

    @property
    def version(self):
        return 0

    @property
    def experiment(self):
        return None

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        pass

    @rank_zero_only
    def log_image(self, key, images, step=None, **kwargs):
        for index, image in enumerate(images):
            path = self.directory / key / f"{index:02d}_{step:06d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            Image.fromarray(np.asarray(image)).save(path)
