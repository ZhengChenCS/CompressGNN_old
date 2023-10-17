import torch
import sys
sys.path.append('../../layer')
from KPropagate import KPropagate
from torch import Tensor
import torch.nn.functional as F
from CompressgnnCluster import Compressgnn_Cluster
from CompressgnnReconstruct import Compressgnn_Reconstruct
from metric import batch_distance
from typing import Dict, List, Optional, Tuple, Union

class GCN(torch.nn.Module):
    def __init__(self, in_features, out_features, param_H=1):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_features=in_features, out_features=in_features, bias=True)
        self.P1 = KPropagate()
        self.P2 = KPropagate()
        self.lin2 = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        self.cluster1 = Compressgnn_Cluster(in_feature=in_features, param_H=param_H, training=True, cache=True, index_cache=False)
        self.cluster2 = Compressgnn_Cluster(in_feature=in_features, param_H=param_H, training=True, cache=True, index_cache=False)
        self.reconstruct = Compressgnn_Reconstruct()
    
    def reset_cache(self):
        self.cluster1.reset_cache()
        self.cluster2.reset_cache()

    def forward(self, x: Tensor, edge_index, edge_weight=None, vertex_cnt:int=0, rule_cnt:int=0) -> Tensor:
        x = self.P1(x=x, edge_index=edge_index, edge_weight=edge_weight, vertex_cnt=vertex_cnt, rule_cnt=rule_cnt)
        x, index1 = self.cluster1(x)
        x = self.lin1(x).relu()
        x = self.reconstruct(x, index1)
        x = self.P2(x=x, edge_index=edge_index, edge_weight=edge_weight, vertex_cnt=vertex_cnt, rule_cnt=rule_cnt)
        x, index2 = self.cluster2(x)
        x = self.lin2(x) 
        x = self.reconstruct(x, index2)
        out = F.log_softmax(x, dim=1) 
        return x
    
    def inference(self, x: Tensor, edge_index, edge_weight=None, vertex_cnt:int=0, rule_cnt:int=0) -> Tensor:
        x = self.P1(x=x, edge_index=edge_index, edge_weight=edge_weight, vertex_cnt=vertex_cnt, rule_cnt=rule_cnt)
        x = self.lin1(x).relu()
        x = self.P2(x=x, edge_index=edge_index, edge_weight=edge_weight, vertex_cnt=vertex_cnt, rule_cnt=rule_cnt)
        x = self.lin2(x) 
        out = F.log_softmax(x, dim=1) 
        return x
