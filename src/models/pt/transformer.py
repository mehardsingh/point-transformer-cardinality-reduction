from pointnet_util import index_points, square_distance
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k
        
    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)
        
        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f
        
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f
        
        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre
        return res, attn
    
class TOMETransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_points)
        self.top_k = k
        
    def forward(self, features):
        # print("========")
        # print(features.shape)
        # print(features.shape)
        x = self.fc1(features)
        # print(x.shape)
        query, key, value = self.w_qs(x), self.w_ks(x), self.w_vs(x)

        # attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(key.size(-1)))
        # attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        # attention_output = torch.matmul(attention_weights, value)

        # print(query.shape)

        attention_scores = torch.bmm(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(key.size(-1)))
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_output = torch.bmm(attention_weights, value)

        # print(attention_output.shape)

    #    # Compute attention scores
    #     scores = torch.matmul(query.unsqueeze(1), key.transpose(-2, -1)).squeeze(1)
    #     # Select top-k values
    #     _, top_k_indices = scores.topk(self.top_k, dim=-1)
    #     # Use indices to gather corresponding value vectors
    #     expanded_indices = top_k_indices.unsqueeze(-1).expand(-1, -1, value.size(-1))

    #     value = torch.gather(value, dim=1, index=expanded_indices)

    #     # Compute attention weights using the top-k scores
    #     attention_weights = torch.nn.functional.softmax(top_k_scores, dim=-1)

    #     # Compute attention output by applying attention weights to value vectors
    #     attention_output = torch.matmul(attention_weights.unsqueeze(1), value).squeeze(1)

        res = self.fc2(attention_output) + features

        # print(res.shape)
        res = self.layer_norm(res)

        return res