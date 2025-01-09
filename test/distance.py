import torch
import sys
sys.path.append('../loader')
sys.path.append('../layer')
from origin_data import Data
from metric import batch_distance
from SLPropagate import SLPropagate


# data = torch.load("/home/chenzheng/CompressGNN/CompressGNNDataset/dataset/Cora/origin/data_coo.pt")
# data = torch.load("/home/chenzheng/CompressGNN/CompressGNNDataset/dataset/CiteSeer/origin/data_coo.pt")
data = torch.load("/home/chenzheng/CompressGNN/CompressGNNDataset/dataset/PubMed/origin/data_coo.pt")
x = data.x
edge_index = data.edge_index
number = []
# print(batch_distance(x))
number.append(batch_distance(x).numpy().item())

SLPropagate = SLPropagate()

for i in range(10):
    x = SLPropagate(x, edge_index)
    # print(batch_distance(x))
    number.append(batch_distance(x).numpy().item())
print(number)