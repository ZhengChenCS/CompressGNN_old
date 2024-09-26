import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
import shutil
import logging
import time
import argparse
import numpy as np
import sys
from tqdm import tqdm
sys.path.append("../loader/")
from origin_data import Data
from neighbor_loader import NeighborLoader
from torch_geometric.nn import SAGEConv

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', filename="gcn_nocache.record", filemode='a') # include timestamp
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamps

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help="Input graph data")
parser.add_argument('--device', type=str, default='cpu', help="device")
args = parser.parse_args()
torch.autograd.set_detect_anomaly(True)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dataset = torch.load(args.data)
    logger.info("Dataset=%s", args.data)
    print(dataset)
    data = dataset
    kwargs = {'batch_size': 1, 'num_workers': 1, 'persistent_workers': True}
    train_loader = NeighborLoader(data, input_nodes=data.train_mask,
                              num_neighbors=[25, 10], shuffle=True, **kwargs)
    
    for batch in train_loader:
        print(batch)
        break
