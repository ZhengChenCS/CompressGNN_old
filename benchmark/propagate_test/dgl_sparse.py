import torch
import dgl
import dgl.sparse as dglsp
from torch import Tensor
import os
import sys
sys.path.append("../../loader")
from origin_data import Data
import time
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def spmm(A, B:Tensor):
    return dglsp.spmm(A, B)


def main():
    root_path = "/home/KongmingDataset"
    dataset = sys.argv[1]
    ori_path = os.path.join(root_path, dataset, "origin/data_coo.pt")
    coo_data = torch.load(ori_path)
    sp_mat = dglsp.spmatrix(coo_data.edge_index, coo_data.edge_weight)
    sp_mat = sp_mat.T
    mat = coo_data.x

    print("Origin Data ...")
    print("=======================================================")
    device = torch.device("cpu")
    sp_mat = sp_mat.to(device)
    mat = mat.to(device)
    t_start = time.perf_counter()
    for i in range(10):
        out = spmm(sp_mat, mat)
    t_end = time.perf_counter()
    print("CPU platform for 10 run_times: {:.4f}s".format(t_end-t_start))

    device = torch.device("cuda")
    sp_mat = sp_mat.to(device)
    mat = mat.to(device)
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    for i in range(100):
        out = spmm(sp_mat, mat)
    t_end = time.perf_counter()
    print("GPU platform for 100 run_times: {:.4f}s".format(t_end-t_start))
    
    print("Compress Data...")
    print("=======================================================")
    ori_path = os.path.join(root_path, dataset, "compress/data_coo_3.pt")
    coo_data = torch.load(ori_path)
    sp_mat = dglsp.spmatrix(coo_data.edge_index, coo_data.edge_weight)
    mat = coo_data.x
    weight = coo_data.edge_weight
    rule_mat = torch.rand((coo_data.rule_cnt, mat.size()[1]), dtype=mat.dtype, device=mat.device)
    mat = torch.cat((mat, rule_mat), dim=0)

    device = torch.device("cpu")
    sp_mat = sp_mat.to(device)
    mat = mat.to(device)
    t_start = time.perf_counter()
    for i in range(10):
        out = spmm(sp_mat, mat)
    t_end = time.perf_counter()
    print("CPU platform for 10 run_times: {:.4f}s".format(t_end-t_start))

    device = torch.device("cuda")
    sp_mat = sp_mat.to(device)
    mat = mat.to(device)
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    for i in range(100):
        out = spmm(sp_mat, mat)
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    print("GPU platform for 100 run_times: {:.4f}s".format(t_end-t_start))


if __name__ == "__main__":
    main()