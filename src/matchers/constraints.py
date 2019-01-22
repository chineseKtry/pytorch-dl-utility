import torch

from . import WeightMatcher

class MaxNormConstraint(WeightMatcher):
    def __init__(self, max_norm, **args):
        super(MaxNormConstraint, self).__init__(**args)
        self.max_norm = max_norm
    
    def function(self, module):
        param = module.weight
        norm = param.norm(2, dim=0, keepdim=True)
        desired = torch.clamp(norm, 0, self.max_norm)
        param = param * (desired / (norm + 1e-8))
