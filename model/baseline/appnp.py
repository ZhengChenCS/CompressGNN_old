import torch
import sys
sys.path.append('../../kongming_layer')
import SLPropagate
from SLPropagate import SLPropagate
from torch import Tensor
import torch.nn.functional as F
from metric import cosine_distance

class APPNP(torch.nn.Module):
    def __init__(self, in_features, out_features, K:int, alpha:float):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_features=in_features, out_features=in_features)
        self.lin2 = torch.nn.Linear(in_features=in_features, out_features=out_features)
        self.P = SLPropagate(add_self_loops=False, normalize=False)
        self.K = K
        self.alpha = alpha

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.lin1(x)
        x = self.lin2(x)
        h = x
        for k in range(self.K):
            x = self.P(x, edge_index)
            x = x * (1 - self.alpha)
            x += self.alpha * h
        out = F.log_softmax(x, dim=1)  # [N, out_c]
        return x
