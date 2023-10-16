import torch
import sys
sys.path.append('../../kongming_layer')
import PUPropagate
from PUPropagate import PUPropagate
from KongmingCluster import Kongming_Cluster
from KongmingReconstruct import Kongming_Reconstruct
from torch import Tensor
import torch.nn.functional as F
from metric import cosine_distance



class SGC(torch.nn.Module):
    def __init__(self, in_features, out_features, K, param_H=1):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=in_features, out_features=out_features)
        self.P = PUPropagate(cached=False, K=K, normalize=True)
        self.cluster = Kongming_Cluster(in_feature=in_features, param_H=param_H, training=True)
        self.reconstruct = Kongming_Reconstruct()
    
    def reset_cache(self):
        self.P.reset_parameters()
        
    def propagate(self, x, edge_index):
        return self.P(x, edge_index)
    
    def transform(self, x):
        x, index = self.cluster(x)
        # print(x.size()[0])
        x = self.lin(x)
        x = self.reconstruct(x, index)
        out = F.log_softmax(x, dim=1)  # [N, out_c]
        return out

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.propagate(x, edge_index)
        x = self.transform(x)
        return x
