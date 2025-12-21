import torch
from einops import rearrange, repeat
from ....geometry.projection import get_world_rays
from ....geometry.projection import sample_image_grid
import torch.nn.functional as F
import MinkowskiEngine as ME



def project_features_to_me(intrinsics, extrinsics, out, depth, voxel_resolution, b, v):
    device = out.device
  
    h, w = depth.shape[2:]
    _, c, _, _ = out.shape
    
    intrinsics = rearrange(intrinsics, "b v i j -> b v () () () i j")
    extrinsics = rearrange(extrinsics, "b v i j -> b v () () () i j")
    depths = rearrange(depth, "b v h w -> b v (h w) () ()")

    uv_grid = sample_image_grid((h, w), device)[0]
    uv_grid = repeat(uv_grid, "h w c -> 1 v (h w) () () c", v=v)
    origins, directions = get_world_rays(uv_grid, extrinsics, intrinsics)
    world_coords = origins + directions * depths[..., None]
    world_coords = world_coords.squeeze(3).squeeze(3)  # [B, V, N, 3]

    features = rearrange(out, "(b v) c h w -> b v c h w", b=b, v=v)
    features = rearrange(features, "b v c h w -> b v h w c")
    features = rearrange(features, "b v h w c -> b v (h w) c")  # [B, V, N, C]
    
    all_points = rearrange(world_coords, "b v n c -> (b v n) c")  # [B*V*N, 3]
    feats_flat = features.reshape(-1, c)  # [B*V*N, C]
    
    with torch.no_grad():
        quantized_coords = torch.round(all_points / voxel_resolution).long()

        # Create coordinate matrix: batch index + quantized coordinates
        batch_indices = torch.arange(b, device=device).repeat_interleave(v * h * w).unsqueeze(1)
        combined_coords = torch.cat([batch_indices, quantized_coords], dim=1)

        # Get unique voxel IDs and mapping indices
        unique_coords, inverse_indices, counts = torch.unique(
            combined_coords,
            dim=0,
            return_inverse=True,
            return_counts=True
        )
    
    num_voxels = unique_coords.shape[0]

    aggregated_feats = torch.zeros(num_voxels, c, device=device)
    aggregated_feats.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, c), feats_flat)
    aggregated_feats = aggregated_feats / counts.view(-1, 1).float()  # Average features

    aggregated_points = torch.zeros(num_voxels, 3, device=device)
    aggregated_points.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, 3), all_points)
    aggregated_points = aggregated_points / counts.view(-1, 1).float()

    # Use correct coordinate format: batch index + quantized coordinates
    sparse_tensor = ME.SparseTensor(
        features=aggregated_feats,
        coordinates=unique_coords.int(),
        tensor_stride=1,
        device=device
    )
    
    return sparse_tensor, aggregated_points, counts