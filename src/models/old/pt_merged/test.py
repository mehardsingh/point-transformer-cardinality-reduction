from point_transformer_cls import get_model
import torch

device = "cpu"

model = get_model(num_class=40, input_dim=3).to(device)
data = torch.rand(32, 1024, 3).to(device)

out = model(data)
print(out.shape)

# device = "mps"

# node_idx = torch.torch.randint(low=0, high=10, size=(32, 10)).to(device)
# src_idx = torch.torch.randint(low=0, high=10, size=(32, 10, 1)).to(device)

# # dst_idx_1 = node_idx[..., None].gather(dim=-2, index=src_idx)
# dst_idx_2 = node_idx.gather(dim=-1, index=src_idx.squeeze(-1)).unsqueeze(-1)

# # print(torch.all(torch.eq(dst_idx_1, dst_idx_2)))
