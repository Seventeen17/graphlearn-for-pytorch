import graphlearn_torch as glt
import torch

indices = torch.tensor([0, 1, 0, 1])
indptr = torch.tensor([0, 2, 4])
values = torch.tensor([0.1, 0.1, 0.1, 0.1])

A = torch.tensor([[0.1, 0.1], [0.1, 0.1]], dtype=torch.float)
ret = glt.spmm(indptr, indices, values, {2, 2}, A)
print(ret)
