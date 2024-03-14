from point_transformer_cls import get_model
import torch
import sys

sys.path.append("src/downsamples")
from fps_knn import FPS_KNN_Downsample
from tome import TOME_Downsample

device = "cpu"

downsample = FPS_KNN_Downsample
model = get_model(downsample, num_class=40, input_dim=3).to(device)
data = torch.rand(32, 1024, 3).to(device)

out = model(data)
print(out.shape)

device = "cpu"

downsample = TOME_Downsample
model = get_model(downsample, num_class=40, input_dim=3).to(device)
data = torch.rand(32, 1024, 3).to(device)

out = model(data)
print(out.shape)
