from point_transformer_cls import get_model
import torch
import sys

sys.path.append("src/train")
from config import Config

device = "cpu"

x = torch.rand(32, 1024, 3).to(device)

config = Config(
    method="tome_xyz", 
    num_points=1024, 
    num_class=10, 
    input_dim=3, 
    init_hidden_dim=32, 
    k=16
)
model = get_model(config).to(device)

pred = model(x)
print(pred.shape)