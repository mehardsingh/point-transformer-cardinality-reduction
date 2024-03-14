from point_transformer_cls import get_model
import torch
import sys

sys.path.append("src/downsamples/fps_knn_pct")
from fps_knn_pct import FPS_KNN_PCT

sys.path.append("src/downsamples/tome")
from tome import TOME

# device = "cpu"

# downsample = FPS_KNN_Downsample
# model = get_model(downsample, num_class=40, input_dim=3).to(device)
# data = torch.rand(32, 1024, 3).to(device)

# out = model(data)
# print(out.shape)

device = "cpu"

downsample = FPS_KNN_PCT

model = get_model(1024, 10, 3, 64, downsample, 32).to(device)
data = torch.rand(32, 1024, 3).to(device)

out = model(data)
print(out.shape)
