# source: https://github.com/qq456cvb/Point-Transformers/blob/master/models/Menghao/model.py

import torch
import torch.nn as nn
from fps_knn_pct import FPS_KNN_PCT
import sys
import numpy as np

sys.path.append("src/tome")
from tome import TOME

sys.path.append("src/random_subsample")
from random_subsample import Random_Subsample_Feature

# default initial hidden dim = 64

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
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x

class PCT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        output_channels = cfg.num_class
        d_points = cfg.input_dim
        self.conv1 = nn.Conv1d(d_points, cfg.init_hidden_dim, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(cfg.init_hidden_dim, cfg.init_hidden_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(cfg.init_hidden_dim)
        self.bn2 = nn.BatchNorm1d(cfg.init_hidden_dim)
        
        if cfg.method == "tome":
            self.downsample1 = TOME(npoint=cfg.num_points//2, in_channels=cfg.init_hidden_dim, out_channels=2*cfg.init_hidden_dim)
            self.downsample2 = TOME(npoint=cfg.num_points//4, in_channels=2*cfg.init_hidden_dim, out_channels=4*cfg.init_hidden_dim)
        elif cfg.method == "normal":
            self.downsample1 = FPS_KNN_PCT(npoint=cfg.num_points//2, nsample=cfg.k, in_channels=cfg.init_hidden_dim, out_channels=2*cfg.init_hidden_dim)
            self.downsample2 = FPS_KNN_PCT(npoint=cfg.num_points//4, nsample=cfg.k, in_channels=2*cfg.init_hidden_dim, out_channels=4*cfg.init_hidden_dim)
        else:
            self.random_subsample1 = Random_Subsample_Feature(npoint=cfg.num_points//2, in_channels=cfg.init_hidden_dim, out_channels=2*cfg.init_hidden_dim)
            self.random_subsample2 = Random_Subsample_Feature(npoint=cfg.num_points//4, in_channels=2*cfg.init_hidden_dim, out_channels=4*cfg.init_hidden_dim)

        self.pt_last = StackedAttention(channels=4*cfg.init_hidden_dim)

        self.relu = nn.ReLU()
        
        self.conv_fuse = nn.Sequential(nn.Conv1d(20*cfg.init_hidden_dim, 16*cfg.init_hidden_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(16*cfg.init_hidden_dim),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(16*cfg.init_hidden_dim, 8*cfg.init_hidden_dim, bias=False)
        self.bn6 = nn.BatchNorm1d(8*cfg.init_hidden_dim)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(8*cfg.init_hidden_dim, 4*cfg.init_hidden_dim)
        self.bn7 = nn.BatchNorm1d(4*cfg.init_hidden_dim)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(4*cfg.init_hidden_dim, output_channels)

        self.cfg = cfg

    def forward(self, x):
        """
        In
        """
        xyz = x[..., :3]
        x = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N
        x = x.permute(0, 2, 1) # B, N, D

        if self.cfg.method == "tome":
            feature_0 = self.downsample1(x)
            feature_1 = self.downsample2(feature_0)
            feature_1 = feature_1.permute(0, 2, 1)
        elif self.cfg.method == "normal":
            new_xyz, feature_0 = self.downsample1(xyz, x)
            new_xyz, feature_1 = self.downsample2(new_xyz, feature_0)
            feature_1 = feature_1.permute(0, 2, 1)
        else:
            feature_0 = self.random_subsample1(x)
            feature_1 = self.random_subsample2(feature_0)
            feature_1 = feature_1.permute(0,2,1)
        
        x = self.pt_last(feature_1)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        x = torch.max(x, 2)[0]
        x = x.view(batch_size, -1)

        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)

        return x

def get_model(cfg):
    return PCT(cfg)

def get_loss():
    return torch.nn.CrossEntropyLoss()
    