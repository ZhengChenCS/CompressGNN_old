import torch
import os
import sys
sys.path.append("../loader")
from origin_data import Data
from CompressgnnData import CompressgnnData
sys.path.append("../layer")
import argparse
import numpy as np 
from torch import Tensor
from torch_scatter import scatter
from compressgnn_runtime import spmm_func
from CompressgnnSPMM import compress_spmm
from torch_sparse import SparseTensor, matmul


def spmm(A: SparseTensor, B: Tensor):
    vlist, elist, value = A.csr()
    return matmul(A, B, reduce="add")

ori_dataset = torch.load("/mnt/disk1/GNNDataset/Reddit/origin/data_csr.pt")
compress_dataset = torch.load("/mnt/disk1/GNNDataset/Reddit/compress/data_csr_8_gorder.pt")
order = np.load("order.npy")

print(ori_dataset)
print(compress_dataset)

ori_dataset = ori_dataset.to("cuda")
compress_dataset = compress_dataset.to("cuda")

compress_out = compress_spmm(
            compress_dataset.edge_index,
            compress_dataset.x,
            )
compress_out = compress_spmm(compress_dataset.edge_index, compress_out)

compress_out = compress_out[order]


out = spmm(ori_dataset.edge_index, ori_dataset.x)
out = spmm(ori_dataset.edge_index, out)

# print(compress_out)
# print(out)

is_close = torch.allclose(out, compress_out, rtol=1e-3, atol=1e-4)
print(is_close)


