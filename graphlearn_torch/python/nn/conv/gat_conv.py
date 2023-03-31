# Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

# TODO 1: implement ours
import dgl.sparse as dglsp

# TODO 2: The base class, which is the program model
class GATConv(nn.Module):
  def __init__(self, in_size, out_size, num_heads, dropout):
        super().__init__()

        self.out_size = out_size
        self.num_heads = num_heads

        self.dropout = nn.Dropout(dropout)
        self.W = nn.Linear(in_size, out_size * num_heads)
        self.a_l = nn.Parameter(torch.zeros(1, out_size, num_heads))
        self.a_r = nn.Parameter(torch.zeros(1, out_size, num_heads))
        self.reset_parameters()

  def reset_parameters(self):
      gain = nn.init.calculate_gain("relu")
      nn.init.xavier_normal_(self.W.weight, gain=gain)
      nn.init.xavier_normal_(self.a_l, gain=gain)
      nn.init.xavier_normal_(self.a_r, gain=gain)

  def forward(self, A_hat, Z):
        Z = self.dropout(Z) # [2708, 1433]
        Z = self.W(Z).view(Z.shape[0], self.out_size, self.num_heads) #[2708, 8, 8]

        # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
        e_l = (Z * self.a_l).sum(dim=1) #[2708, 8]
        e_r = (Z * self.a_r).sum(dim=1) #[2708, 8]
        e = e_l[A_hat.row] + e_r[A_hat.col] # [10556, 8] +  [10556, 8] -> [10556, 8]
        # e = dglsp.sddmm(A_hat, e_l, e_r)

        a = F.leaky_relu(e)
        A_atten = dglsp.val_like(A_hat, a).softmax() # reshpae edges with scores as shape of A
        a_drop = self.dropout(A_atten.val)
        A_atten = dglsp.val_like(A_atten, a_drop)
        return dglsp.bspmm(A_atten, Z)

class GAT(torch.nn.Module):
    def __init__(
        self, in_size, out_size, hidden_size=8, num_heads=8, dropout=0.6
    ):
        super().__init__()

        self.in_conv = GATConv(
            in_size, hidden_size, num_heads=num_heads, dropout=dropout
        )
        self.out_conv = GATConv(
            hidden_size * num_heads, out_size, num_heads=1, dropout=dropout
        )

    def forward(self, A_hat, X):
        # Flatten the head and feature dimension.
        Z = F.elu(self.in_conv(A_hat, X)).flatten(1)
        # Average over the head dimension.
        Z = self.out_conv(A_hat, Z).mean(-1)
        return Z

class GAT(torch.nn.Module):
    def __init__(
        self, in_size, out_size, hidden_size=8, num_heads=8, dropout=0.6
    ):
        super().__init__()

        self.in_conv = GATConv(
            in_size, hidden_size, num_heads=num_heads, dropout=dropout
        )
        self.out_conv = GATConv(
            hidden_size * num_heads, out_size, num_heads=1, dropout=dropout
        )

    def forward(self, A_hat, X):
        # Flatten the head and feature dimension.
        Z = F.elu(self.in_conv(A_hat, X)).flatten(1)
        # Average over the head dimension.
        Z = self.out_conv(A_hat, Z).mean(-1)
        return Z

#################################################################################
# TODO(wenting.swt): Move into examples.
import os.path as osp
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.logging import log

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, 'Cora', transform=T.NormalizeFeatures())
g = dataset[0].to(device)

# Create the sparse adjacency matrix A.
N = g.num_nodes
A = dglsp.spmatrix(g.edge_index, shape=(N, N))

# Add self-loops.
I = dglsp.identity(A.shape, device=device)
A_hat = A + I

# Create GAT model.
X = g.x

model = GAT(dataset.num_features, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(A_hat, X)
    loss = F.cross_entropy(out[g.train_mask], g.y[g.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred =  model(A_hat, X).argmax(dim=-1)

    accs = []
    for mask in [g.train_mask, g.val_mask, g.test_mask]:
        accs.append(int((pred[mask] == g.y[mask]).sum()) / int(mask.sum()))
    return accs

import time
t1 = time.perf_counter()
best_val_acc = final_test_acc = 0
for epoch in range(1, 200 + 1):
    loss = train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
torch.cuda.synchronize()
t2 = time.perf_counter()
print(t2 - t1)

# Epoch: 200, Loss: 0.7951, Train: 0.9929, Val: 0.8120, Test: 0.8270, 2.5s
