import torch
import torch.nn as nn

from matchers import WeightMatcher

class L1Regularizer(WeightMatcher):
    def __init__(self, l1, **args):
        super(L1Regularizer, self).__init__(**args)
        self.l1 = l1

    def function(self, module):
        return self.l1 * torch.norm(module.weight, 1)
