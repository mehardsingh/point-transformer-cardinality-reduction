import torch.nn as nn
from src.models.pt.pointnet_util import PointNetSetAbstraction

class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)
        
    def forward(self, xyz, points):
        x = self.sa(xyz, points)
        return x
    
class FPS_KNN_PT(nn.Module):
    def __init__(self, npoint, nsample, in_channels, out_channels):
        super().__init__()
        self.td = TransitionDown(npoint, nsample, [in_channels, out_channels, out_channels])

    def forward(self, xyz, points):
        return self.td(xyz, points)