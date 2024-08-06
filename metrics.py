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


class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes: int, margin: float = 0.1, scale: float = 1.0) -> None:
        super().__init__()
        self.s = scale
        self.n_classes = num_classes
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)

    def forward(self, cosine: Tensor, labels: Tensor) -> Tensor:
        positve = F.one_hot(labels, self.n_classes) == 1
        above_thrs = cosine > self.threshold
        use_margin = torch.logical_and(positve, above_thrs)

        sine = torch.sqrt(1.0 - torch.pow(cosine[use_margin], 2))
        cosine[use_margin] = cosine[use_margin] * self.cos_m - sine * self.sin_m
        
        loss = F.cross_entropy(self.s * cosine, labels) 
        return loss