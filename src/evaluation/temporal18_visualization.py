"""Export voxel Gaussians directly, without reshaping them onto an image grid."""

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

from ..model.ply_export import construct_list_of_attributes


def save_gaussians(gaussians, batch_index, path):
    means = gaussians.means[batch_index].detach().cpu().numpy()
    covariance = gaussians.covariances[batch_index].detach().cpu().numpy()
    eigenvalues, rotation = np.linalg.eigh(covariance)
    rotation[:, :, 0] *= np.linalg.det(rotation)[:, None]
    quaternion = Rotation.from_matrix(rotation).as_quat()[:, [3, 0, 1, 2]]
    scales = np.sqrt(np.maximum(eigenvalues, np.finfo(np.float32).tiny))
    sh = gaussians.harmonics[batch_index].detach().cpu().numpy()
    opacity = gaussians.opacities[batch_index].detach().cpu().numpy()
    opacity = np.clip(opacity, np.finfo(np.float32).eps, 1 - np.finfo(np.float32).eps)
    rest = sh[:, :, 1:].reshape(len(means), -1)
    values = np.concatenate((means, np.zeros_like(means), sh[:, :, 0], rest,
                             np.log(opacity / (1 - opacity))[:, None], np.log(scales), quaternion), axis=1)
    elements = np.empty(len(means), dtype=[(name, "f4") for name in construct_list_of_attributes(rest.shape[1])])
    elements[:] = list(map(tuple, values))
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)
