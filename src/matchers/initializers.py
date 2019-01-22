import torch
import torch.nn as nn

from . import WeightMatcher


class KaimingNormalInitializer(WeightMatcher):
    def function(self, module):
        nn.init.kaiming_normal_(module.weight.data, **self.args)


class ConstantInitializer(WeightMatcher):
    def __init__(self, weight, **args):
        super(ConstantInitializer, self).__init__(**args)
        self.weight = weight

    def function(self, module):
        nn.init.constant_(module.weight, self.weight)