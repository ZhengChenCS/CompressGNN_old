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
sys.path.append("../../../kongming_layer/")
sys.path.append("../../../loader/")
sys.path.append("../../../Application")
from KongmingData import KongmingData
from sgc_loader import SGCData, SGCDataLoader
from baseline.sgc import SGC

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', filename="gcn_nocache.record", filemode='a') # include timestamp
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamps

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help="Input graph data")
parser.add_argument('--num_features', type=int, required=True, help="Input num of features")
parser.add_argument('--num_classes', type=int, required=True, help="Input num of classes")
parser.add_argument('--epochs', type=int, required=True, help="Input num of epoch")
parser.add_argument('--device', type=str, default='cpu', help="device")
args = parser.parse_args()
torch.autograd.set_detect_anomaly(True)


def test(model, data, device):
    model = model.to(device)
    data = data.to(device)
    model.eval()
    _, prediction = model.inference(data.x, data.edge_index).max(dim=1)
    # _, prediction = model(data.x, data.edge_index).max(dim=1)
    target = data.y.squeeze()

    test_correct = prediction[data.test_mask].eq(target[data.test_mask]).sum().item()
    test_number = data.test_mask.sum().item()
    logger.info("Accuracy of Test Samples: {:.2f}%".format( (test_correct / test_number * 100) ))
    print("Accuracy of Test Samples: {:.2f}%".format( (test_correct / test_number * 100) ))

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
    old = 100
    patent = 0
    num_epoch = 0
    for epoch in tqdm(range(epochs)):
        pred = model(data.x, data.edge_index)
        loss = F.cross_entropy(pred[mask], y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if old < loss:
            patent += 1
        else:
            if (old - loss)  < 1e-5 and epoch != 0:
                break
            old = loss
        if patent > 5:
            print("patent > 5")
            break
        num_epoch += 1

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    print("Epoch: {:d}".format(num_epoch))
    return t_end - t_start



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dataset = torch.load(args.data)
    logger.info("Dataset=%s", args.data)
    # logger.info("Num of features=%s", args.num_features)
    # logger.info("Num of classes=%s", args.num_classes)
    dataset.x = dataset.x.to(torch.float32)
    param_H = int(math.log(dataset.x.size()[0], 2))+1

    model = SGC(in_features=args.num_features, out_features=args.num_classes, K=2)

    if args.device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    warm_up_time = train_runtime(model, dataset, epochs=1, device=device)
    
    logger.info("Train time for {:d} epoch: {:.4f}s".format(10, warm_up_time))
    
    model.reset_cache()

    train_time = train_runtime(model, dataset, epochs=args.epochs, device=device)
    logger.info("Train time for {:d} epoch: {:.4f}s".format(args.epochs, train_time))
    print("Propagate time:{:.4f}".format(model.p_time))
    print("Train time for {:d} epoch: {:.4f}s".format(args.epochs, train_time))

    test(model, dataset, device)
    