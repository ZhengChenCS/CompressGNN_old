from typing import Optional, Tuple
import torch
from torch_sparse.tensor import SparseTensor
import numpy as np
import sys
sys.path.append("../loader")
from CompressgnnData import CompressgnnData
from compress_graph import CompressGraph 
import time
from compressgnn_sample import neighbor_sample_ori_impl, neighbor_sample_cg_sim


def neighbor_sample(src: SparseTensor, subset: torch.Tensor, num_neighbor,
               replace: bool = False, is_directed: bool = True) -> Tuple[SparseTensor, torch.Tensor]:

    rowptr, col, value = src.csr()
    node, row, col = neighbor_sample_ori_impl(rowptr, col, subset, num_neighbors)

    return node, row, col

def cgnn_neighbor_sample(src: CompressGraph, subset: torch.Tensor, num_neighbors,
               replace: bool = False, is_directed: bool = True) :
    v2v_vlist, v2v_elist, _ = src.v2v_graph.csr()
    v2r_vlist, v2r_elist, _ = src.v2r_graph.csr()
    r2v_vlist, r2v_elist, _ = src.r2v_graph.csr()
    r2r_vlists = []
    r2r_elists = []
    for r2r_graph in src.r2r_graph:
        r2r_vlist, r2r_elist, _ = r2r_graph.csr()
        r2r_vlists.append(r2r_vlist)
        r2r_elists.append(r2r_elist)
    v_degree = src.vertex_degree
    r_degree = src.rule_degree
    v_node, v_row, v_col = neighbor_sample_cg_sim(v2v_vlist, v2v_elist, 
                                                v2r_vlist, v2r_elist,
                                                r2v_vlist, r2v_elist,
                                                r2r_vlists, r2r_elists,
                                                v_degree, r_degree,
                                                subset, num_neighbors)
    

    return v_node, v_row, v_col


if __name__ == "__main__":
    # data_path = "/home/chenzheng/CompressGNN/CompressGNNDataset/dataset/uk-2007-05@1000000/compress/data_csr_3.pt"
    data_path = "/home/chenzheng/CompressGNN/CompressGNNDataset/dataset/cnr-2000/compress/data_csr_3.pt"
    data = torch.load(data_path)
    cg = data.edge_index
    print(cg)
    num_nodes = data.vertex_cnt
    print(num_nodes)
    batch_size = 1024
    # num_neighbors = [25, 10, 10]
    num_neighbors = [25]
    p = np.random.permutation(num_nodes)
    pos = 0
    start = time.perf_counter()
    while pos < num_nodes - batch_size:
        subset = p[pos:pos+batch_size]
        # subset = np.array([272878])
        subset = torch.from_numpy(subset)
        node, row, col = cgnn_neighbor_sample(cg, subset, num_neighbors)
        pos += batch_size
        # break
    subset = p[pos:num_nodes]
    subset = torch.from_numpy(subset)
    node, row, col = cgnn_neighbor_sample(cg, subset, num_neighbors)
    end = time.perf_counter()
    print("Sample time: {:.2f}s".format( (end-start)))

