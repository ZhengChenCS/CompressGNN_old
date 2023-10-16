import os
import sys
sys.path.append("../../loader")
sys.path.append("../../kongming_layer")
import torch
from typing import List
import time
from origin_data import Data
import torch_sparse
from torch_sparse import SparseTensor, matmul
from torch import Tensor
from compress_graph import CompressGraph
from KongmingData import KongmingData
from kongming import spmm_func
from KongmingSPMM import compress_spmm
import numpy as np
from torch_scatter import scatter

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def compare(A: Tensor, B: Tensor):
    A = A.cpu()
    B = B.cpu()
    indices = ~np.isclose(A, B, rtol=1e-3)
    row, col = np.where(indices)
    print(row)
    print(row.shape)
    return np.allclose(A, B, rtol=1e-6)
        

def spmm(A: SparseTensor, B: Tensor):
    vlist, elist, value = A.csr()
    if not B.is_cuda:
        return spmm_func(vlist, elist, value , B, "rowbalance", None)
    else:
        return spmm_func(vlist, elist, value , B, "nnzbalance_rowcache", None)

def propagate(src: Tensor, dst: Tensor, weight:Tensor,  B:Tensor):
    edge_content = B.index_select(-2, src)
    edge_content = weight.view(-1, 1) * edge_content
    return scatter(edge_content, dst, dim=-2, dim_size=B.size()[0], reduce="add")

def compress_propagate(edge_index: List[Tensor], weight:Tensor, B:Tensor, vertex_cnt:int, rule_cnt:int):
    out = torch.zeros(
        (vertex_cnt + rule_cnt, B.size()[1]), dtype=B.dtype, device=B.device
    )
    step = len(edge_index)
    for i in range(step):
        if i == 0:
            src = B.index_select(-2, edge_index[i][0])
        else:
            src = out.index_select(-2, edge_index[i][0])
        src = weight[i].view(-1, 1) * src
        out = scatter(src, edge_index[i][1], dim=-2, dim_size=vertex_cnt+rule_cnt, reduce="add", out=out)
    return out[:vertex_cnt]  



def main():
    root_path = "/home/KongmingDataset"
    dataset = sys.argv[1]

    print("Origin Data ...")
    print("=======================================================")
    ori_path = os.path.join(root_path, dataset, "origin/data_coo.pt")
    coo_data = torch.load(ori_path)
    sp_mat = SparseTensor(
        row=coo_data.edge_index[1],
        col=coo_data.edge_index[0],
        value=coo_data.edge_weight)
    mat = coo_data.x

    device = torch.device("cuda")
    sp_mat = sp_mat.to(device)
    mat = mat.to(device)
    coo_data = coo_data.to(device)
    origin_csr_out = spmm(sp_mat, mat)
    origin_coo_out = propagate(coo_data.edge_index[0], coo_data.edge_index[1], coo_data.edge_weight, mat)
    compare(origin_csr_out, origin_coo_out)
    

    print("Compress Data...")
    print("=======================================================")
    ori_path = os.path.join(root_path, dataset, "compress/data_csr_3.pt")
    coo_path = os.path.join(root_path, dataset, "compress/data_coo_3.pt")
    csr_data = torch.load(ori_path)
    coo_data = torch.load(coo_path)

    device = torch.device("cuda")
    coo_data = coo_data.to(device)
    csr_data = csr_data.to(device)
    
    compress_csr_out = compress_spmm(
        csr_data.edge_index,
        csr_data.x)
    compress_coo_out = compress_propagate(coo_data.edge_index, coo_data.edge_weight, coo_data.x, coo_data.vertex_cnt, coo_data.rule_cnt)
    compare(compress_coo_out, compress_csr_out)


if __name__ == "__main__":
    main()
