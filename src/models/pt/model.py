import torch
import torch.nn as nn
from transformer import TransformerBlock, TOMETransformerBlock, MyAttention, MyAttention2
from fps_knn_pt import FPS_KNN_PT
import sys
import numpy as np

sys.path.append("src/tome")
from tome import TOME

sys.path.append("src/random_subsample")
from random_subsample import Random_Subsample_XYZ

# default initial hidden dim = 32

class Backbone(nn.Module):
    def __init__(self, cfg, nblocks=4):
        super().__init__()
        # npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.fc1 = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.init_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.init_hidden_dim, cfg.init_hidden_dim)
        )

        if cfg.method in ["normal", "random"]:
            self.transformer1 = TransformerBlock(cfg.init_hidden_dim, 16*cfg.init_hidden_dim, cfg.k)
        else:
            # print(cfg.init_hidden_dim, 16*cfg.init_hidden_dim, cfg.k)
            self.transformer1 = MyAttention2(cfg.init_hidden_dim, 16*cfg.init_hidden_dim, cfg.k)

        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()

        for i in range(nblocks):
            channel = cfg.init_hidden_dim * 2 ** (i + 1)
            if cfg.method == "normal":
                self.transition_downs.append(FPS_KNN_PT(cfg.num_points // 4 ** (i + 1), cfg.k, channel // 2 + 3, channel))
                self.transformers.append(TransformerBlock(channel, 16*cfg.init_hidden_dim, cfg.k))
            elif cfg.method == "tome_ft":
                self.transition_downs.append(TOME(cfg.num_points // 4 ** (i + 1), channel // 2, channel, use_xyz=False))
                self.transformers.append(MyAttention2(channel, 16*cfg.init_hidden_dim, cfg.k))
            elif cfg.method == "tome_xyz":
                self.transition_downs.append(TOME(cfg.num_points // 4 ** (i + 1), channel // 2, channel, use_xyz=True))
                self.transformers.append(MyAttention2(channel, 16*cfg.init_hidden_dim, cfg.k))
            elif cfg.method == "random":
                self.transition_downs.append(Random_Subsample_XYZ(cfg.num_points // 4 ** (i + 1), channel // 2, channel))
                self.transformers.append(TransformerBlock(channel, 16*cfg.init_hidden_dim, cfg.k))
        
        self.nblocks = nblocks
        self.cfg = cfg
    
    def forward(self, x):
        xyz = x[..., :3]
        
        if self.cfg.method == "normal":
            points = self.transformer1(xyz, self.fc1(x))[0]
            xyz_and_feats = [(xyz, points)]
            for i in range(self.nblocks):
                xyz, points = self.transition_downs[i](xyz, points)
                points = self.transformers[i](xyz, points)[0]
                xyz_and_feats.append((xyz, points))
        
            return points, xyz_and_feats
        
        elif self.cfg.method in ["tome_ft", "tome_xyz"]:
            points = self.transformer1(self.fc1(x))
            for i in range(self.nblocks):
                points, xyz = self.transition_downs[i](points, xyz)
                points = self.transformers[i](points)

            return points
        
        else:
            points = self.transformer1(xyz, self.fc1(x))[0]
            xyz_and_feats = [(xyz, points)]
            for i in range(self.nblocks):
                points, xyz = self.transition_downs[i](points, xyz)
                points = self.transformers[i](xyz, points)[0]
                xyz_and_feats.append((xyz, points))
        
            return points, xyz_and_feats


class PointTransformerCls(nn.Module):
    def __init__(self, cfg, nblocks=4):
        super().__init__()
        self.backbone = Backbone(cfg)
        # npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.fc2 = nn.Sequential(
            nn.Linear(cfg.init_hidden_dim * 2 ** nblocks, 8*cfg.init_hidden_dim),
            nn.ReLU(),
            nn.Linear(8*cfg.init_hidden_dim, 2*cfg.init_hidden_dim),
            nn.ReLU(),
            nn.Linear(2*cfg.init_hidden_dim, cfg.num_class)
        )
        self.nblocks = nblocks
        self.cfg = cfg
    
    def forward(self, x):
        if self.cfg.method in ["normal", "random"]:
            points, _ = self.backbone(x)
        else:
            points = self.backbone(x)

        res = self.fc2(points.mean(1))
        return res

def get_model(cfg):
    return PointTransformerCls(cfg)

def get_loss():
    return torch.nn.CrossEntropyLoss()