import torch
import torch.nn as nn
from tome_utils import bipartite_soft_matching

class Merge(nn.Module):
    def __init__(self, use_xyz, npoint, compress=False):
        super().__init__()
        self.npoint = npoint
        self.compress = compress
        self.use_xyz = use_xyz

    def forward(self, points, xyz):
        r = points.shape[1] - self.npoint
        pmerger, punmerger, xyzmerger, xyzunmerger = bipartite_soft_matching(points, xyz, r)
        merged = pmerger(points)

        if not self.use_xyz: 
            return merged, None, None
        
        else:
            merged_xyz = xyzmerger(xyz)
            if self.compress:
                compressed_xyz = xyzunmerger(merged_xyz)    
                return merged, merged_xyz, compressed_xyz
            else:
                return merged, merged_xyz, None

class TOME(nn.Module):
    def __init__(self, use_xyz, npoint, in_channels, out_channels, compress=False):
        super().__init__()
        self.merge = Merge(use_xyz, npoint, compress=compress)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.use_xyz = use_xyz

    def forward(self, points, xyz):
        merged, merged_xyz, compressed_xyz = self.merge(points, xyz) # merged: B, N, in_channels
        merged = merged.permute(0, 2, 1) #  B, in_channels, N
        merged = self.relu(self.bn(self.conv(merged))) # B, out_channels, N
        merged = merged.permute(0, 2, 1) # B, N, out_channels

        return merged, merged_xyz, compressed_xyz