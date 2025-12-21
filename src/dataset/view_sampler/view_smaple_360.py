'''
Modifiedy from latentSplat and pixelSplat to handle extrapolate and more context views
'''

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from jaxtyping import Float, Int64
from torch import Tensor
import random

from .view_sampler import ViewSampler


@dataclass
class ViewSampler360UniformCfg:
    name: Literal["360_uniform"]
    num_context_views: int
    num_target_views: int
    min_angular_separation: int = 10  # Minimum angular separation (degrees)
    context_view_strategy: Literal["equidistant", "farthest_point"] = "equidistant"


class ViewSampler360Uniform(ViewSampler[ViewSampler360UniformCfg]):
    def get_camera_angles(self, extrinsics: Tensor) -> Float[Tensor, "view"]:
        """Calculate the azimuth angle of each camera in world coordinates (0-360 degrees)"""
        camera_positions = extrinsics[:, :3, 3]
        # Calculate vectors relative to the scene center
        centroid = camera_positions.mean(dim=0)
        vectors = camera_positions - centroid
        # Calculate azimuth angle (arctan2 handles 360 degree range)
        azimuth = torch.atan2(vectors[:, 1], vectors[:, 0])  # -pi to pi
        return (torch.rad2deg(azimuth) % 360 + 360) % 360    # Convert to 0-360 degrees

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> tuple[
        Int64[Tensor, " context_view"], 
        Int64[Tensor, " target_view"],
    ]:
        num_views = extrinsics.shape[0]
        angles = self.get_camera_angles(extrinsics).cpu().numpy()

        # ===== Core Improvement 1: 360-degree uniform context view selection =====
        if self.cfg.context_view_strategy == "equidistant":
            # Equal angular interval sampling
            target_angles = torch.linspace(0, 360, self.cfg.num_context_views + 1)[:-1]
            context_indices = []
            for angle in target_angles:
                diffs = torch.abs(torch.tensor(angles) - angle.item())
                diffs = torch.min(diffs, 360 - diffs)  # Handle circular distance
                context_indices.append(torch.argmin(diffs).item())

        else:  # farthest_point
            # Farthest point sampling in 3D space
            camera_pos = extrinsics[:, :3, 3].unsqueeze(0)  # [1, views, 3]
            context_indices = farthest_point_sample(camera_pos, self.cfg.num_context_views)[0]

        # ===== Core Improvement 2: Maximize dispersion of target views =====
        all_indices = set(range(num_views))
        remaining_indices = list(all_indices - set(context_indices))

        if len(remaining_indices) >= self.cfg.num_target_views:
            # Maximize angular separation from remaining views
            candidate_angles = [angles[i] for i in remaining_indices]
            target_indices = []
            
            for _ in range(self.cfg.num_target_views):
                max_min_diff = -1
                best_idx = None
                
                for idx, angle in zip(remaining_indices, candidate_angles):
                    current_angles = [angles[i] for i in context_indices + target_indices] + [angle]
                    min_diff = min(
                        min(abs(a - b) for a, b in zip(current_angles[:-1], current_angles[1:])),
                        360 - max(current_angles) + min(current_angles)
                    )
                    
                    if min_diff > max_min_diff:
                        max_min_diff = min_diff
                        best_idx = idx
                
                if best_idx is not None:
                    target_indices.append(best_idx)
                    remaining_indices.remove(best_idx)
                    candidate_angles.remove(angles[best_idx])
        else:
            # Random selection when insufficient views available
            target_indices = random.sample(range(num_views), self.cfg.num_target_views)

        return (
            torch.tensor(sorted(context_indices)),  # Keep sorted for easier debugging
            torch.tensor(target_indices),
        )


    # Maintain original property implementation
    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views