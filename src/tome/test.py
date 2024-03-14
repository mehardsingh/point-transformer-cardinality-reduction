import torch
from tome_utils import bipartite_soft_matching

batch_size = 32
num_points = 1024
hidden_dim = 32


xyz = torch.rand(batch_size, num_points, 3)
features = torch.rand(batch_size, num_points, 32)

merge_fn, unmerge_fn, merge_xyz_fn, unmerge_xyz_fn = bipartite_soft_matching(features, xyz, r=num_points//2)

merged = merge_fn(features)
unmerged = unmerge_fn(merged)

merged_xyz = merge_fn(xyz)
unmerged_xyz = unmerge_fn(merged_xyz)

print(merged.shape)
print(unmerged.shape)
print(merged_xyz.shape)
print(unmerged_xyz.shape)