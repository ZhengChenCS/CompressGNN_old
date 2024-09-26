from typing import Optional, Tuple

import torch
from torch_sparse.tensor import SparseTensor
import numpy as np
import sys
sys.path.append("../loader")
from CompressgnnData import CompressgnnData
from compress_graph import CompressGraph 
import time


def neighbor_sample(src: SparseTensor, subset: torch.Tensor, num_neighbor,
               replace: bool = False, is_directed: bool = True) -> Tuple[SparseTensor, torch.Tensor]:

    rowptr, col, value = src.csr()

    node, row, col, edge = torch.ops.torch_sparse.neighbor_sample(
        rowptr, col, subset, num_neighbors, replace, is_directed)


    return node, row, col, edge

def cgnn_neighbor_sample(src: CompressGraph, subset: torch.Tensor, num_neighbor,
               replace: bool = False, is_directed: bool = True) -> Tuple[SparseTensor, torch.Tensor]:

    v2v_rowptr, v2v_col, _ = src.v2v_graph.csr()
    v2r_rowptr, v2r_col, _ = src.v2r_graph.csr()
    r2r_rowptr, r2r_col, _ = src.r2r_graph[0].csr()
    r2v_rowptr, r2v_col, _ = src.r2v_graph.csr()
    # rowptr, col, value = src.csr()
    v_node, v_row, v_col, v_edge = torch.ops.torch_sparse.neighbor_sample(
        v2v_rowptr, v2v_col, subset, num_neighbors, replace, is_directed
    )
    r_node, r_row, r_col, r_edge = torch.ops.torch_sparse.neighbor_sample(
        v2r_rowptr, v2r_col, subset, num_neighbors, replace, is_directed
    )
    print(r_node)
    
    
    


    # node, row, col, edge = torch.ops.torch_sparse.neighbor_sample(
    #     rowptr, col, subset, num_neighbors, replace, is_directed)


    return v_node, v_row, v_col, v_edge


if __name__ == "__main__":
    # data_path = "/home/chenzheng/CompressGNN/CompressGNNDataset/dataset/uk-2007-05@1000000/compress/data_csr_3.pt"
    data_path = "/home/chenzheng/CompressGNN/CompressGNNDataset/dataset/cnr-2000/compress/data_csr_3.pt"
    data = torch.load(data_path)
    adj = data.edge_index
    num_nodes = data.vertex_cnt
    batch_size = 1024
    num_neighbors = [25, 10, 10]
    # num_neighbors = [25]
    p = np.random.permutation(num_nodes)
    pos = 0
    start = time.perf_counter()
    while pos < num_nodes - batch_size:
        subset = p[pos:pos+batch_size]
        subset = torch.from_numpy(subset)
        node, row, col, edge = cgnn_neighbor_sample(adj, subset, num_neighbors)
        pos += batch_size
        break
    # subset = p[pos:num_nodes]
    # subset = torch.from_numpy(subset)
    # node, row, col, edge = cgnn_neighbor_sample(adj, subset, num_neighbors)
    end = time.perf_counter()
    print("Sample time: {:.2f}s".format( (end-start)))

