import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import math
from torch_geometric.loader import DataLoader
import shutil
import logging
import time
import argparse
import numpy as np
import sys
from tqdm import tqdm
sys.path.append("../../kongming_layer/")
sys.path.append("../../loader")
sys.path.append("../../Application/kongming/")
from KongmingData import KongmingData
from sgc_loader import SGCData, SGCDataLoader

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', filename="gcn_nocache.record", filemode='a') # include timestamp
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamps


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn', help="models used")
parser.add_argument('--data', type=str, required=True, help="Input graph data")
parser.add_argument('--is_saved', type=bool, default=False, help="Whether to save model parameter.")
parser.add_argument('--is_test', type=bool, default=False, help="Whether to test accuracy of test samples.")
parser.add_argument('--model_path', type=str, default="", help="Model saved path")
parser.add_argument('--num_features', type=int, required=True, help="Input num of features")
parser.add_argument('--num_classes', type=int, required=True, help="Input num of classes")
# parser.add_argument('--num_hid', type=int, required=True, help="Input num of hidden units")
parser.add_argument('--epoch', type=int, required=True, help="Input num of epoch")
parser.add_argument('--device', type=str, default='cpu', help="device")
parser.add_argument('--sgc_path', type=str, help="sgc save path for propagate result")
args = parser.parse_args()
torch.autograd.set_detect_anomaly(True)

def test(model, data, device):
    model = model.to(device)
    data = data.to(device)
    model.eval()
    _, prediction = model.inference(data.x, data.edge_index, data.edge_mask, data.vertex_cnt, data.rule_cnt).max(dim=1)
    # _, prediction = model(data.x, data.edge_index).max(dim=1)

    target = data.y.squeeze()

    test_correct = prediction[data.test_mask].eq(target[data.test_mask]).sum().item()
    test_number = data.test_mask.sum().item()

    logger.info("Accuracy of Test Samples: {:.2f}%".format( (test_correct / test_number * 100) ))

def train_runtime(model, data, epochs, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model = model.to(device)
    data = data.to(device)
    mask = data.train_mask.to(device)
    model.train()
    y = data.y[mask].squeeze()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    if data.graph_type == "coo":
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_modules = True
        ) as prof:
            for epoch in tqdm(range(epochs)):
                pred = model(data.x, data.edge_index, data.edge_weight, data.vertex_cnt, data.rule_cnt)
                loss = F.cross_entropy(pred[mask], y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    elif data.graph_type == "csr":
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_modules = True
        ) as prof:
            for epoch in tqdm(range(epochs)):
                pred = model(data.x, data.edge_index, row_map=data.row_map, col_map=data.col_map, spmm_attr=data.spmm_attr, vertex_cnt=data.vertex_cnt, rule_cnt=data.rule_cnt)
                loss = F.cross_entropy(pred[mask], y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    prof.export_chrome_trace("./kongming_gcn_profile.json")

    return t_end - t_start

def sgc_propagate(model, data, device):
    model = model.to(device)
    data = data.to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    p_start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        x = model.propagate(data.x, data.edge_index, data.edge_mask, data.vertex_cnt, data.rule_cnt)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    p_end = time.perf_counter()
    dataset = SGCData(x=x, y=data.y, train_mask=data.train_mask, vaild_mask=data.vaild_mask, test_mask=data.test_mask)
    if not os.path.exists(args.sgc_path):
        torch.save(dataset, args.sgc_path)
    return p_end-p_start

def sgc_transform(model, data, epochs, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model =  model.to(device)
    data = torch.load(args.sgc_path)
    data = data.to(device)
    mask = data.train_mask.to(device)
    model.train()
    y = data.y[mask].squeeze()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    for epoch in tqdm(range(epochs)):
        pred = model.transform(data.x)
        loss = F.cross_entropy(pred[mask], y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    return t_end-t_start


def train_sgc(model, data, epochs, device):
    p_time = sgc_propagate(model, data, device)
    t_time = sgc_transform(model, data, epochs, device)
    logger.info("propagate time: {:.4f}s".format(p_time))
    logger.info("transform time: {:.4f}s".format(t_time))
    return p_time + t_time

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # dicide wichi GPU to use
    # cora_dataset = get_data()
    dataset = torch.load(args.data)
    print(dataset)
    print(dataset.edge_index)
    logger.info("Dataset=%s", args.data)
    logger.info("Num of features=%s", args.num_features)
    logger.info("Num of classes=%s", args.num_classes)
    # logger.info("Num of hidden units=%s", args.num_hid)
    # print(dataset[0].edge_index)
    dataset.x = dataset.x.to(torch.float32)
    param_H = int(math.log(dataset.x.size()[0], 2))+1
    # param_H = 30

    # dataset[0].x = dataset[0].x.to(torch.float32)

    if args.model == "gcn":
        from gcn import GCN 
        my_net = GCN(in_features=args.num_features, out_features=args.num_classes, param_H=param_H)
    elif args.model == "sgc":
        from sgc import SGC
        my_net = SGC(in_features=args.num_features, out_features=args.num_classes, K=2, param_H=param_H)
    elif args.model == "appnp":
        from appnp import APPNP
        my_net = APPNP(in_features=args.num_features, out_features=args.num_classes, K=2, alpha=0.85, param_H=param_H)
    
    if args.device == 'cpu':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Wram up for GPU
    if args.model == "sgc":
        wram_up_time = train_sgc(my_net, dataset, epochs=1, device=device)
    else:
        wram_up_time = train_runtime(my_net, dataset, epochs=10, device=device)
        pass
    logger.info("Wram up  time elapsed: {:.4f}s".format(wram_up_time))
    
    if args.model == "sgc":
        my_net.reset_cache()

    if args.model == "sgc":
        train_time = train_sgc(my_net, dataset, epochs=args.epoch, device=device)
    else:
        train_time = train_runtime(my_net, dataset, epochs=args.epoch, device=device)

    logger.info("Train time for {:d} epoch: {:.4f}s".format(args.epoch, train_time))
    return
    if args.is_saved:
        torch.save(my_net.state_dict(), args.model_path)
    
    if args.is_test:
        test(my_net, dataset, device=device)



if __name__ == '__main__':
    main()