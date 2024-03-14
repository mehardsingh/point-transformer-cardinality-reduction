import torch
import torch.nn as nn
from pointnet_util import farthest_point_sample, index_points, square_distance

class Sample_Group(nn.Module):
    def __init__(self, npoint, nsample):
        super().__init__()
        self.npoint = npoint
        self.nsample = nsample

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.npoint 
        
        fps_idx = farthest_point_sample(xyz, self.npoint) # [B, npoint]

        new_xyz = index_points(xyz, fps_idx) 
        new_points = index_points(points, fps_idx)

        dists = square_distance(new_xyz, xyz)  # B x npoint x N
        idx = dists.argsort()[:, :, :self.nsample]  # B x npoint x K

        grouped_points = index_points(points, idx)
        grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
        new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, self.nsample, 1)], dim=-1)
        return new_xyz, new_points

class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N
        x = torch.max(x, 2)[0]
        x = x.view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x
    
class FPS_KNN_PCT(nn.Module):
    def __init__(self, npoint, nsample, in_channels, out_channels):
        super().__init__()
        self.sample_and_group = Sample_Group(npoint, nsample)
        self.local_op = Local_op(in_channels*2, out_channels)

    def forward(self, xyz, points):
        new_xyz, new_points = self.sample_and_group(xyz, points)
        new_points = self.local_op(new_points) # B, C, N
        new_points = new_points.permute(0, 2, 1)
        return new_xyz, new_points
