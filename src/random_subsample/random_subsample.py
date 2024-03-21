import torch
import torch.nn as nn
import numpy as np

class Random_Subsample_Feature(nn.Module):
    def __init__(self, npoint, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.npoint = npoint

    def forward(self, points):
        indices = np.random.choice(points.shape[1], self.npoint, replace=False)
        new_points = points[:, indices, :].permute(0, 2, 1)
        new_points = self.relu(self.bn(self.conv(new_points))).permute(0, 2, 1)
        
        return new_points
    
class Random_Subsample_XYZ(nn.Module):
    def __init__(self, npoint, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.npoint = npoint

    def forward(self, points, xyz):
        indices = np.random.choice(points.shape[1], self.npoint, replace=False)
        new_points = points[:, indices, :].permute(0, 2, 1)
        new_xyz = xyz[:, indices, :]
        new_points = self.relu(self.bn(self.conv(new_points))).permute(0, 2, 1)
        
        return new_points, new_xyz