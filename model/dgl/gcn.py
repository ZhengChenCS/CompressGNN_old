import torch
import sys
from torch import Tensor
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import time
import dgl.sparse as dglsp
from torch_sparse import SparseTensor
from type import OptTensor


class Propagate(torch.nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, x: Tensor, graph: SparseTensor):
        vlist, elist, value = graph.csr()
        sp_mat = dglsp.from_csr(
            indptr=vlist,
            indices=elist,
            val=value
        )
        x = dglsp.spmm(sp_mat, x)
        return x


class GCN(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_features=in_features, out_features=in_features, bias=True)
        self.P1 = Propagate()
        self.P2 = Propagate()
        self.lin2 = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=True)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.P1(x, edge_index)
        x = self.lin1(x).relu()
        x = self.P2(x, edge_index)
        x = self.lin2(x)
        x = F.log_softmax(x, dim=1)  # [N, out_c]
        return x

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.P1(x, edge_index)
        x = self.lin1(x).relu()
        x = self.P2(x, edge_index)
        x = self.lin2(x)
        x = F.log_softmax(x, dim=1)  # [N, out_c]
        return x



