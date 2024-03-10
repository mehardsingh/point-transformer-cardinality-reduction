# source: https://github.com/qq456cvb/Point-Transformers/blob/master/models/Menghao/model.py

import torch
import torch.nn as nn
from pointnet_util import farthest_point_sample, index_points, square_distance
import sys

sys.path.append("src/merge_utils")
from merge_fb import bipartite_soft_matching

def merge(points, npoint):
    # B, N, C = xyz.shape
    # S = npoint 
    
    # fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint]

    # new_xyz = index_points(xyz, fps_idx) 
    # new_points = index_points(points, fps_idx)

    # dists = square_distance(new_xyz, xyz)  # B x npoint x N
    # idx = dists.argsort()[:, :, :nsample]  # B x npoint x K

    # grouped_points = index_points(points, idx) # B x npoint x K x C
    # grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)  # maintains shift invariance 
    # # append global information
    # new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    # # new_points
    # #  is B x npoint x K x 2*C 

    r = points.shape[1] - npoint
    pmerger, punmerger = bipartite_soft_matching(points, r)
    merged = pmerger(points)
    compressed = punmerger(merged)    
    return merged, compressed

class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        b, n, s, d = x.size()  # batch, npoint ,K ,  C 
        x = x.permute(0, 1, 3, 2) # B x npoint, c , k
        x = x.reshape(-1, d, s)  # (B x npoint) , C , K 
        batch_size, _, N = x.size()

        # apply linear layer to each point 
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N

        # Take max over neighbors (why not avg?)
        x = torch.max(x, 2)[0]
        x = x.view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1) # B, n, OutChanels
        return x


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv(x)# b, c, n        
        x_v = self.v_conv(x)
        energy = x_q @ x_k # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = x_v @ attention # b, c, n 
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
    

class StackedAttention(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()

        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))

        x1 = self.sa1(x)
        # our stuff 
        x2 = self.sa2(x1)
        # our stuff 
        x3 = self.sa3(x2)
        # our stuff 
        x4 = self.sa4(x3)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x


class PointTransformerCls(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        output_channels = cfg.num_class
        d_points = cfg.input_dim
        self.conv1 = nn.Conv1d(d_points, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        self.pt_last = StackedAttention()

        self.relu = nn.ReLU()
        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        """
        In
        """

        # ENCODE the point cloud into (BxNxD)
        # print("===Encode===")
        x = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N
        x = x.permute(0, 2, 1) # B, N, D
        # print("x", x.shape)
        # print()

        # print("===Bipartite Soft Match===")
        feature_0_merged, feature_0_compressed = merge(x, npoint=512)    
        feature_0_merged = feature_0_merged.permute(0, 2, 1)  
        feature_0_merged = self.relu(self.bn3(self.conv3(feature_0_merged)))
        feature_0_merged = feature_0_merged.permute(0, 2, 1) 
        # print(feature_0_merged.shape)
        # print()

        # print("===Bipartite Soft Match===")
        feature_1_merged, feature_1_compressed = merge(feature_0_merged, npoint=256)  
        feature_1_merged = feature_1_merged.permute(0, 2, 1)  
        feature_1_merged = self.relu(self.bn4(self.conv4(feature_1_merged)))
        # print(feature_1_merged.shape)

        # print()
        
        # print("===Stacked Attention===")
        x = self.pt_last(feature_1_merged)
        x = torch.cat([x, feature_1_merged], dim=1)
        x = self.conv_fuse(x)
        # print(x.shape)
        # print()

        # print("===Maxpool===")
        x = torch.max(x, 2)[0]
        x = x.view(batch_size, -1)
        # print(x.shape)
        # print()

        # print("===CLS Head===")
        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)

        return x

class Config:
    def __init__(self, num_class, input_dim):
        self.num_class = num_class
        self.input_dim = input_dim

def get_model(num_class=40, input_dim=3):
    return PointTransformerCls(Config(num_class=num_class, input_dim=input_dim))

def get_loss():
    return torch.nn.CrossEntropyLoss()

    