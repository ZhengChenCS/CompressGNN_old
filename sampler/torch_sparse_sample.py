from typing import Optional, Tuple

import torch
from torch_sparse.tensor import SparseTensor
import numpy as np
import sys
sys.path.append("../loader")
from origin_data import Data
import time


def neighbor_sample(src: SparseTensor, subset: torch.Tensor, num_neighbor,
               replace: bool = False, is_directed: bool = True) -> Tuple[SparseTensor, torch.Tensor]:

    rowptr, col, value = src.csr()

    node, row, col, edge = torch.ops.torch_sparse.neighbor_sample(
        rowptr, col, subset, num_neighbors, replace, is_directed)


    return node, row, col, edge


if __name__ == "__main__":
    data_path = "/mnt/disk1/KongmingDataset/uk-2007-05@1000000/origin/data_csr.pt"
    # data_path = "/mnt/disk1/KongmingDataset/cnr-2000/origin/data_csr.pt"
    data = torch.load(data_path)
    print(data)
    adj = data.edge_index
    num_nodes = data.vertex_cnt
    batch_size = 1024
    subset = torch.tensor([0, 1])
    num_neighbors = [25, 10, 10]
    p = np.random.permutation(num_nodes)
    pos = 0
    start = time.perf_counter()
    while pos < num_nodes - batch_size:
        subset = p[pos:pos+batch_size]
        subset = torch.from_numpy(subset)
        node, row, col, edge = neighbor_sample(adj, subset, num_neighbors)
        pos += batch_size
        # print(pos)
    subset = p[pos:num_nodes]
    subset = torch.from_numpy(subset)
    node, row, col, edge = neighbor_sample(adj, subset, num_neighbors)
    end = time.perf_counter()
    print("Sample time: {:.2f}s".format( (end-start)))

