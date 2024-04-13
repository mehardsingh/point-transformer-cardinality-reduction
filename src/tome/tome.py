import torch
import torch.nn as nn
from tome_utils import bipartite_soft_matching

class Merge(nn.Module):
    def __init__(self, npoint, compress=False, use_xyz=False):
        super().__init__()
        self.npoint = int(npoint)
        self.compress = compress
        self.use_xyz = use_xyz

    def forward(self, points, xyz):
        r = points.shape[1] - self.npoint
        # pmerger, punmerger, xyzmerger, xyzunmerger = bipartite_soft_matching(points, r)
        if self.use_xyz:
            pmerger, punmerger = bipartite_soft_matching(xyz, r)
        else:
            pmerger, punmerger = bipartite_soft_matching(points, r)

        merged_pts = pmerger(points)
        merged_xyz = pmerger(xyz)

        compressed_pts = None
        compressed_xyz = None
        if self.compress:
            compressed_pts = punmerger(merged_pts)    
            compressed_xyz = punmerger(merged_xyz)    

        return merged_pts, compressed_pts, merged_xyz, compressed_xyz
    
class TOME(nn.Module):
    def __init__(self, npoint, in_channels, out_channels, compress=False, use_xyz=False):
        super().__init__()
        self.use_xyz = use_xyz
        self.merge = Merge(npoint, compress=compress, use_xyz=use_xyz)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, points, xyz):
        merged_pts, compressed_pts, merged_xyz, compressed_xyz = self.merge(points, xyz) # merged: B, N, in_channels

        merged_pts = merged_pts.permute(0, 2, 1) #  B, in_channels, N
        merged_pts = self.relu(self.bn(self.conv(merged_pts))) # B, out_channels, N
        merged_pts = merged_pts.permute(0, 2, 1) # B, N, out_channels

        return merged_pts, merged_xyz
    
