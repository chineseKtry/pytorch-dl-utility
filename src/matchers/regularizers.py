import torch
import torch.nn as nn

from . import WeightMatcher
from ..util.computing import numpy_to_builtin 

class L1Regularizer(WeightMatcher):
    def __init__(self, l1, **args):
        super(L1Regularizer, self).__init__(**args)
        self.l1 = numpy_to_builtin(l1)

    def function(self, module):
        return self.l1 * torch.norm(module.weight, 1)
