from point_transformer_cls import PointTransformerCls
import torch

class Config:
    def __init__(self, num_class, input_dim):
        self.num_class = num_class
        self.input_dim = input_dim

cfg = Config(num_class=10, input_dim=3)
model = PointTransformerCls(cfg)

data = torch.rand(64, 1024, 3)

print(model)

print(model(data).shape)