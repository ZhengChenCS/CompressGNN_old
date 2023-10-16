import sys
sys.path.append('../../kongming_layer/')
from SLPropagate import SLPropagate
from KongmingCluster import Kongming_Cluster
from KongmingReconstruct import Kongming_Reconstruct
from torch import Tensor
from metric import cosine_distance,batch_distance
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import math
import logging
import argparse
sys.path.append("../../loader")
from KongmingData import KongmingData
from sgc_loader import SGCData, SGCDataLoader

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', filename="gcn_nocache.record", filemode='a') # include timestamp
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamps


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help="Input graph data")
parser.add_argument('--num_features', type=int, required=True, help="Input num of features")
parser.add_argument('--num_classes', type=int, required=True, help="Input num of classes")
parser.add_argument('--log_path', type=str, default="log.txt", help="log path")
parser.add_argument('--device', type=str, default='cpu', help="device")
args = parser.parse_args()
logger.addHandler(logging.FileHandler(args.log_path))

class TEST_MODEL(torch.nn.Module):
    def __init__(self, in_features, out_features, param_H=1):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_features=in_features, out_features=in_features)
        self.P1 = SLPropagate()
        # self.P2 = SLPropagate()
        # self.lin2 = torch.nn.Linear(in_features=in_features, out_features=out_features)
        self.cluster = Kongming_Cluster(in_feature=in_features, param_H=param_H, training=True, cache=True)
        self.reconstruct = Kongming_Reconstruct()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.P1(x, edge_index)
        cosine = batch_distance(x)
        logger.info(f"cosine: {cosine}")
        better = (1-cosine)*x.size()[0]
        before = x.size()[0]
        logger.info(f"better be {better}")
        logger.info(f"before cluster {before}")
        x, index = self.cluster(x)
        after = x.size()[0]
        logger.info(f"after cluster {after}")
        logger.info(f"There should be {(better-after)/after*100}% percent more")
        x = self.lin1(x)
        x = self.reconstruct(x, index)
        logger.info(x.size())
        return x


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # dicide wichi GPU to use
    # cora_dataset = get_data()
    dataset = torch.load(args.data)
    logger.info(dataset)
    edge_index = dataset.edge_index
    logger.info("Dataset=%s", args.data)
    logger.info("Num of features=%s", args.num_features)
    logger.info("Num of classes=%s", args.num_classes)
    x = dataset.x = dataset.x.to(torch.float32)
    param_H = int(math.log(dataset.x.size()[0], 2))+1

    my_net = TEST_MODEL(in_features=args.num_features, out_features=args.num_classes, param_H=param_H)
    
    if args.device == 'cpu':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_net.to(device)
    x=x.to(device)
    edge_index = edge_index.to(device)
    # Wram up for GPU
    my_net.forward(x = x, edge_index = edge_index)

    return


if __name__ == '__main__':
    main()