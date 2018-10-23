import torch
from fnmatch import fnmatch

class ModuleMatcher(object):
    def __init__(self, module_match):
        self.module_match = module_match
    
    def match_name(self, module_name):
        return fnmatch(module_name, self.module_match)

class MaxNormConstraint(ModuleMatcher):
    def __init__(self, max_norm, module_match):
        super(MaxNormConstraint, self).__init__(module_match)
        self.max_norm = max_norm
    
    def apply(self, module_name, module):
        if self.match_name(module_name) and hasattr(module, 'weight'):
            param = module.weight
            norm = param.norm(2, dim=0, keepdim=True)
            desired = torch.clamp(norm, 0, self.max_norm)
            param = param * (desired / (norm + 1e-8))
