import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ArcFaceHead(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = F.normalize(x, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        return F.linear(x, weight)
    
    def loss(self) -> Tensor:
        """
        Loss to make the space larger between each center
        """
        weight = F.normalize(self.weight, p=2, dim=1)
        n_vec = len(weight)
        loss = 0.
        for i in range(n_vec):
            loss += torch.sum(F.cosine_similarity(weight[i], weight[i+1:]))
        total_n = n_vec * (n_vec - 1) / 2
        loss /= total_n
        return loss


class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes: int, margin: float = 0.1, scale: float = 1.0) -> None:
        super().__init__()
        self.s = scale
        self.n_classes = num_classes
        self.margin = margin
        self.upper = math.cos(margin)
        self.lower = math.cos(math.pi - margin)

    def forward(self, cosine: Tensor, labels: Tensor) -> Tensor:
        positve = F.one_hot(labels, self.n_classes) == 1
        in_range = torch.logical_and(cosine > self.lower, cosine < self.upper)
        use_margin = torch.logical_and(positve, in_range)

        cosine[use_margin] = torch.cos(torch.arccos(cosine[use_margin]) + self.margin)
        
        loss = F.cross_entropy(self.s * cosine, labels) 
        return loss