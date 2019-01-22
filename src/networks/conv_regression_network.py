import torch.nn as nn
import torch.nn.functional as F

class ConvRegressionNetwork(nn.Module):

    def __init__(self, features, regressor):
        super(ConvRegressionNetwork, self).__init__()
        self.features, self.regressor = features, regressor
        self.loss = nn.MSELoss()

    def forward(self, x_t, y_t):
        t = self.features(x_t)
        t = t.view(t.size(0), -1)
        t = self.regressor(t)
        y_pred_t = t[:, 0]

        if y_t is not None:
            loss_t = self.loss(y_pred_t, y_t)
        else:
            loss_t = None
        return loss_t, {
            'loss': loss_t,
            'y': y_pred_t
        }
