import torch
import torch.nn as nn
from tome_utils import bipartite_soft_matching

class Merge(nn.Module):
    def __init__(self, npoint, compress=False):
        super().__init__()
        self.npoint = npoint
        self.compress = compress

    def forward(self, points):
        r = points.shape[1] - self.npoint
        pmerger, punmerger = bipartite_soft_matching(points, r)
        merged = pmerger(points)

        compressed = None
        if self.compress:
            compressed = punmerger(merged)    

        return merged, compressed
    
class TOME(nn.Module):
    def __init__(self, npoint, in_channels, out_channels, compress=False):
        super().__init__()
        self.merge = Merge(npoint, compress=compress)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, points):
        merged, compressed = self.merge(points) # merged: B, N, in_channels
        merged = merged.permute(0, 2, 1) #  B, in_channels, N

        merged = self.relu(self.bn(self.conv(merged))) # B, out_channels, N
        merged = merged.permute(0, 2, 1) # B, N, out_channels

        return merged