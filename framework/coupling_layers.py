import torch
import torch.nn as nn
from .base import BaseCouplingLayer

class AdditiveCouplingLayer(BaseCouplingLayer):
    def __init__(self, coupling_function, chunker):
        super(AdditiveCouplingLayer, self).__init__()
        self.coupling_function = coupling_function
        self.chunker = chunker

    def forward(self, x):
        x1, x2 = self.chunker(x)
        y1 = x1
        y2 = x2 + self.coupling_function(x1)
        return self.chunker.invert(y1, y2)
    
    def inverse(self, y):
        y1, y2 = self.chunker(y)
        x1 = y1
        x2 = y2 - self.coupling_function(x1)
        return self.chunker.invert(x1, x2)
    
    def log_det_jacobian(self, x):
        x1, x2 = self.chunker(x)
        return torch.zeros(x1.size(0), device=x1.device)

class ScalingCouplingLayer(BaseCouplingLayer):
    def __init__(self, dim):
        super(ScalingCouplingLayer, self).__init__()
        self.dim = dim
        self.log_scaling_factors = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        scale = torch.exp(self.log_scaling_factors)
        return x * scale
    
    def inverse(self, y):
        scale = torch.exp(-self.log_scaling_factors)
        return y * scale
    
    def log_det_jacobian(self, x):
        return self.log_scaling_factors.sum(dim=0).repeat(x.size(0))
