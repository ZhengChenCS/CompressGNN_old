import os
import sys
sys.path.append("../../loader")
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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def compress_spmm(graph, x_j):
    v2v_vlist, v2v_elist, v2v_value = graph.v2v_graph.csr()
    v2r_vlist, v2r_elist, v2r_value = graph.v2r_graph.csr()
    r2v_vlist, r2v_elist, r2v_value = graph.r2v_graph.csr()
    # r2r_vlist, r2r_elist, r2r_value = graph.r2r_graph.csr()
    if not x_j.is_cuda:
        rule_out = spmm_func(r2v_vlist, r2v_elist, r2v_value, x_j, "rowbalance", None)
        for i in range(graph.step):
            part_vlist, part_elist, part_value = graph.r2r_graph[i].csr()
            rule_out = spmm_func(part_vlist, part_elist, part_value, rule_out, "rowbalance", rule_out)
        out = spmm_func(v2r_vlist, v2r_elist, v2r_value, rule_out, "rowbalance", None)
        out = spmm_func(v2v_vlist, v2v_elist, v2v_value, x_j, "rowbalance", out)
    else:
        rule_out = spmm_func(r2v_vlist, r2v_elist, r2v_value, x_j, "nnzbalance_rowcache", None)
        for i in range(graph.step):
            part_vlist, part_elist, part_value = graph.r2r_graph[i].csr()
            rule_out = spmm_func(part_vlist, part_elist, part_value, rule_out, "nnzbalance_rowcache", rule_out)
        out = spmm_func(v2r_vlist, v2r_elist, v2r_value, rule_out, "nnzbalance_rowcache", None)
        out = spmm_func(v2v_vlist, v2v_elist, v2v_value, x_j, "nnzbalance_rowcache", out)
    return out
    


def main():
    root_path = "/home/KongmingDataset"
    dataset = sys.argv[1]
    print("Compress Data...")
    print("=======================================================")
    ori_path = os.path.join(root_path, dataset, "compress/data_csr_3.pt")
    csr_data = torch.load(ori_path)
    print(csr_data)

    device = torch.device("cpu")
    csr_data = csr_data.to(device)
    t_start = time.perf_counter()
    for i in range(10):
        out = compress_spmm(
            csr_data.edge_index,
            csr_data.x)
    t_end = time.perf_counter()
    print(out)
    compress_cpu_time = t_end - t_start
    print("CPU platform for 10 run_times: {:.4f}s".format(t_end - t_start))

    device = torch.device("cuda")
    csr_data = csr_data.to(device)
    torch.cuda.synchronize()


    # direct call
    '''
    rule_mat = torch.rand(
        (csr_data.rule_cnt,
         mat.size()[1]),
        dtype=csr_data.x.dtype,
        device=csr_data.x.device)
    mat = torch.cat((csr_data.x, rule_mat), dim=0)
    '''
    t_start = time.perf_counter()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_modules=True
    ) as prof:
        for i in range(100):
            out = compress_spmm(
                csr_data.edge_index,
                csr_data.x)
    torch.cuda.synchronize()
    print(out)
    prof.export_chrome_trace("compress_csr.json")
    t_end = time.perf_counter()
    compress_gpu_time = t_end - t_start
    print("GPU platform for 100 run_times: {:.4f}s".format(t_end - t_start))



if __name__ == "__main__":
    main()
