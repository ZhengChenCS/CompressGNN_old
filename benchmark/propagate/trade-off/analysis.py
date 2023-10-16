import torch 
import os
import sys
sys.path.append("../../../loader")
import time
from KongmingData import KongmingData
from typing import List
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import argparse
import numpy as np 
from torch import Tensor
from torch_scatter import scatter

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help="Input data")
args = parser.parse_args()


if __name__ == '__main__':
    dataset = torch.load(args.data)
    print(dataset)