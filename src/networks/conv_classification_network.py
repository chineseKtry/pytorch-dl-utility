import torch.nn as nn
import torch.nn.functional as F

class ConvClassificationNetwork(nn.Module):

    def __init__(self, features, classifier):
        super(ConvClassificationNetwork, self).__init__()
        self.features, self.classifier = features, classifier
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x_t, y_t):
        t = self.features(x_t)
        t = t.view(t.size(0), -1)
        t = self.classifier(t)
        
        if y_t is not None:
            loss_t = self.loss(t, y_t)
        else:
            loss_t = None
        return loss_t, {
            'loss': loss_t,
            'y': F.softmax(t, dim=1)[:, 1]
        }