import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import shutil
import logging
import time
import argparse
import numpy as np
import sys
from tqdm import tqdm
sys.path.append("../../kongming_layer/")
sys.path.append("../../dataset")
sys.path.append("../../Application/baseline/")
sys.path.append("../../loader/")
from cluster import ClusterData, ClusterLoader
from sgc_loader import SGCData, SGCDataLoader
from neighbor_sampler import NeighborSampler 

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', filename="sample_sgc.record", filemode='a') # include timestamp
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamp

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn', help="models used")
parser.add_argument('--data', type=str, required=True, help="Input graph data")
parser.add_argument('--num_features', type=int, required=True, help="Input num of features")
parser.add_argument('--num_classes', type=int, required=True, help="Input num of classes")
# parser.add_argument('--num_hid', type=int, required=True, help="Input num of hidden units")
parser.add_argument('--epoch', type=int, required=True, help="Input num of epoch")
parser.add_argument('--device', type=str, default='cpu', help="device")
parser.add_argument('--batch_size', type=int, required=True, help="batch size")
parser.add_argument('--num_parts', type=int, required=True, help="cluster_part")
parser.add_argument('--sgc_batch_size', type=int, required=False, help="sgc loader batch size")
parser.add_argument('--save_dir', type=str, required=True, help="save dir for cluster data")
parser.add_argument('--sgc_path', type=str, help="sgc save path for propagate result")
args = parser.parse_args()
# train_loader

# subgraph_loader
num_iter=100

@torch.no_grad()
def test(model, subgraph_loader, device):
    model = model.to(device)
    data = data.to(device)
    model.eval()
    _, prediction = model(data.x, data.edge_index).max(dim=1)

    target = data.y.squeeze()

    test_correct = prediction[data.test_mask].eq(target[data.test_mask]).sum().item()
    test_number = data.test_mask.sum().item()

    # print("Accuracy of Test Samples: ", (test_correct / test_number * 100), '%')
    logger.info("Accuracy of Test Samples: {:.2f}%".format( (test_correct / test_number * 100) ))


def train_runtime(model, train_loader, epochs, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model = model.to(device)
    # data = data.to(device)
    # print(data)
    # mask = data.train_mask.to(device)
    # model.train()
    # y = data.y[mask].squeeze()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    t_start = time.perf_counter()
    
    for batch in train_loader:
        batch = batch.to(device)
        for epoch in tqdm(range(epochs)):
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(pred[batch.train_mask], batch.y[batch.train_mask].squeeze())
            loss.backward()
            optimizer.step()       

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()      

    return t_end - t_start

def sgc_prepare(model, train_loader, device):
    model = model.to(device)
    model.eval()
    x_all = []
    y_all = []
    train_mask_all = []
    vaild_mask_all = []
    test_mask_all = []
    for batch in train_loader:
        batch = batch.to(device)
        x = model.propagate(batch.x, batch.edge_index) 
        x = x.cpu()
        y = batch.y.cpu()
        train_mask = batch.train_mask.cpu()
        vaild_mask = batch.vaild_mask.cpu()
        test_mask = batch.vaild_mask.cpu()
        x_all.append(x)
        y_all.append(y)
        train_mask_all.append(train_mask)
        vaild_mask_all.append(vaild_mask)
        test_mask_all.append(test_mask)
    
    y = torch.cat(y_all, 0)
    x = torch.cat(x_all, 0)
    train_mask = torch.cat(train_mask_all, 0)
    vaild_mask = torch.cat(vaild_mask_all, 0)
    test_mask = torch.cat(test_mask_all, 0)
    dataset = SGCData(x=x, y=y, train_mask=train_mask, vaild_mask=vaild_mask, test_mask=test_mask)
    if not os.path.exists(args.sgc_path):
        torch.save(dataset, args.sgc_path)

def sgc_propagate(model, train_loader, num_iter, device):
    model = model.to(device)
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    p_start = time.perf_counter()
    for batch in train_loader:
        batch = batch.to(device)
        for i in range(num_iter):
            x = model.propagate(batch.x, batch.edge_index) 
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    p_end = time.perf_counter()
    return p_end - p_start

def sgc_transform(model, train_loader, epochs, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    model.train()
    for batch in train_loader:
        batch = batch.to(device)
        for i in range(epochs):
            optimizer.zero_grad()
            pred = model.transform(batch.x)
            loss = F.cross_entropy(pred[batch.train_mask], batch.y[batch.train_mask].squeeze())
            loss.backward()
            optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    return t_end - t_start



def train_sgc(model, train_loader, epochs, device):
    
    p_time = sgc_propagate(model, train_loader, epochs, device)
    logger.info("propagate time: {:.4f}s".format(p_time/epochs))
    if not os.path.exists(args.sgc_path):
        sgc_prepare(model, train_loader, device)
    dataset = torch.load(args.sgc_path)
    if args.sgc_batch_size == -1 or not args.sgc_batch_size :
        args.sgc_batch_size = len(dataset)
    print(args.sgc_batch_size)
    SGCloader = SGCDataLoader(dataset, batch_size=args.sgc_batch_size, shuffle=False, num_workers=12)
    
    t_time = sgc_transform(model, SGCloader, epochs, device)
    logger.info("transform time: {:.4f}s".format(t_time))

    return p_time/epochs + t_time
    

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # dicide wichi GPU to use
    # cora_dataset = get_data()
    dataset = torch.load(args.data)
    logger.info("Dataset=%s", args.data)
    logger.info("Num of features=%s", args.num_features)
    logger.info("Num of classes=%s", args.num_classes)
    # logger.info("Num of hidden units=%s", args.num_hid)
    # print(dataset[0].edge_index)
    dataset.x = dataset.x.to(torch.float32)
    cluster_data = ClusterData(dataset, num_parts=args.num_parts, recursive=False,
                           save_dir=args.save_dir)
    train_loader = ClusterLoader(cluster_data, batch_size=args.batch_size, shuffle=True,
                             num_workers=12)

    subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1], batch_size=1024,
                                  shuffle=False, num_workers=12)
    # dataset[0].x = dataset[0].x.to(torch.float32)

    if args.model == "gcn":
        from sample_gcn import GCN 
        my_net = GCN(in_features=args.num_features, out_features=args.num_classes)
    elif args.model == "sgc":
        from sample_sgc import SGC
        my_net = SGC(in_features=args.num_features, out_features=args.num_classes, K=3)
    elif args.model == "appnp":
        from appnp import APPNP
        my_net = APPNP(in_features=args.num_features, out_features=args.num_classes, K=2, alpha=0.85)

    
    if args.device == 'cpu':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Wram up for GPU
    if args.model == "sgc":
        wram_up_time = train_sgc(my_net, train_loader, epochs=1, device=device)
    else:
        wram_up_time = train_runtime(my_net, train_loader, epochs=1, device=device)
    
    logger.info("Wram up  time elapsed: {:.4f}s".format(wram_up_time))
    return
    if args.model == "sgc":
        my_net.reset_cache()

    if args.model == "sgc":
        train_time = train_sgc(my_net, train_loader, epochs=args.epoch, device=device)
    else:
        train_time = train_runtime(my_net, train_loader, epochs=args.epoch, device=device)
    logger.info("Train time for {:d} epoch: {:.4f}s".format(args.epoch, train_time))


    #model test
    # test(my_net, dataset[0], device=device)
    # test(my_net, dataset, device=device)


if __name__ == '__main__':
    main()