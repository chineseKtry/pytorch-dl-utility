import torch.nn as nn
import torch.nn.functional as F

class ConvRegressionNetwork(nn.Module):

    def __init__(self, features, regressor):
        super(ConvRegressionNetwork, self).__init__()
        self.features, self.regressor = features, regressor

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x[:, 0]