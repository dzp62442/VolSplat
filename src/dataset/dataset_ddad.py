from dataclasses import dataclass
from typing import Literal

from .dataset_temporal18 import DatasetTemporal18, DatasetTemporal18Cfg


@dataclass
class DatasetDDADCfg(DatasetTemporal18Cfg):
    name: Literal["ddad"]


class DatasetDDAD(DatasetTemporal18):
    # The prepared DDAD reference is +X forward, +Y left, +Z up. The
    # nuScenes-trained VolSplat predicts voxel features and Gaussian covariance
    # in a frame with +X right, +Y forward, +Z up. Unlike relative camera
    # geometry, those learned world-space quantities are not rotation invariant.
    # Left-multiply ALL cameras, including their translations, by this change of
    # basis. Camera-local OpenCV axes, metric scale and relative poses stay intact.
    reference_to_model = (
        (0., -1., 0., 0.),
        (1., 0., 0., 0.),
        (0., 0., 1., 0.),
        (0., 0., 0., 1.),
    )

    def _load_views(self, infos):
        views = super()._load_views(infos)
        transform = views["extrinsics"].new_tensor(self.reference_to_model)
        views["extrinsics"] = transform @ views["extrinsics"]
        return views

    def evaluation_metadata(self):
        return {
            **super().evaluation_metadata(),
            "camera_frame": "nuscenes_axes_x_right_y_forward_z_up",
            "reference_to_model": [list(row) for row in self.reference_to_model],
        }
