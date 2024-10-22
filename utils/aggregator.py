import math
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.utils import softmax

class CVAgg(Aggregation):

    def __init__(self, semi_grad: bool = False):
        super().__init__()
        self.semi_grad = semi_grad

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        mean = self.reduce(x, index, ptr, dim_size, dim, reduce='mean')
        if self.semi_grad:
            with torch.no_grad():
                mean2 = self.reduce(x * x, index, ptr, dim_size, dim, 'mean')
                var = mean2 - mean * mean
                std = var.clamp(min=1e-5).sqrt()
                std = std.masked_fill(std <= math.sqrt(1e-5), 0.0)
                cv = std/(mean+1e-8)
        else:
            mean2 = self.reduce(x * x, index, ptr, dim_size, dim, 'mean')
            var = mean2 - mean * mean
            std = var.clamp(min=1e-5).sqrt()
            std = std.masked_fill(std <= math.sqrt(1e-5), 0.0)
            cv = std/(mean+1e-8)
        return cv
