import torch
from torch_scatter import scatter
from torch import Tensor
import os
import sys
sys.path.append("../../loader")
from origin_data import Data
import time
from typing import List
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

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
    src = coo_data.edge_index[0]
    dst = coo_data.edge_index[1]
    weight = coo_data.edge_weight
    mat = coo_data.x

    device = torch.device("cpu")
    src = src.to(device)
    dst = dst.to(device)
    mat = mat.to(device)
    weight = weight.to(device)
    t_start = time.perf_counter()
    for i in range(10):
        out = propagate(src, dst, weight, mat)
    t_end = time.perf_counter()
    print("CPU platform for 10 run_times: {:.4f}s".format(t_end-t_start))

    device = torch.device("cuda")
    src = src.to(device)
    dst = dst.to(device)
    mat = mat.to(device)
    weight = weight.to(device)
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    for i in range(100):
        out = propagate(src, dst, weight, mat)
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    print("GPU platform for 100 run_times: {:.4f}s".format(t_end-t_start))

    print("Compress Data...")
    print("=======================================================")
    ori_path = os.path.join(root_path, dataset, "compress/data_coo_3.pt")
    coo_data = torch.load(ori_path)

    device = torch.device("cpu")
    coo_data = coo_data.to(device)
    t_start = time.perf_counter()
    for i in range(10):
        out = compress_propagate(coo_data.edge_index, coo_data.edge_weight, coo_data.x, coo_data.vertex_cnt, coo_data.rule_cnt)
    t_end = time.perf_counter()
    print("CPU platform for 10 run_times: {:.4f}s".format(t_end-t_start))

    device = torch.device("cuda")
    coo_data = coo_data.to(device)
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    for i in range(100):
        out = compress_propagate(coo_data.edge_index, coo_data.edge_weight, coo_data.x, coo_data.vertex_cnt, coo_data.rule_cnt)
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    print("GPU platform for 100 run_times: {:.4f}s".format(t_end-t_start))
    
    


if __name__ == "__main__":
    main()

