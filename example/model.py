from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from base_model import BaseModel
import util
import metrics
from batch_generator import H5pyBatchGenerator


def get_config():
    return {
        'dropout': np.random.choice([0.1, 0.5, 0.75]),
        'delta': np.random.choice([1e-4, 1e-6, 1e-8]),
        'momentum': np.random.choice([0.9, 0.99, 0.999])
    }

process_x_y = lambda X, Y: (X.astype(np.float32), Y[:, 1].astype(np.long))


def get_train_generator(data_path, batch_size):
    return H5pyBatchGenerator(data_path, 'train', batch_size=batch_size,
                              shuffle=True, process_x_y=process_x_y)


def get_val_generator(data_path, batch_size=None):
    return H5pyBatchGenerator(data_path, 'valid', batch_size=batch_size, process_x_y=process_x_y)


def get_test_generator(data_path, batch_size=None):
    return H5pyBatchGenerator(data_path, 'test', batch_size=batch_size, process_x_y=process_x_y)


def get_model(config, save_dir, args):
    return Model(config, save_dir, args)


class Network(nn.Module):

    def __init__(self, config, gpu):
        super(Network, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 128, (1, 24), padding=(0, 11)),
            nn.ReLU(),
            nn.MaxPool2d((1, 100))
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(32, 2)
        )
        if gpu:
            self.features = self.features.cuda()
            self.classifier = self.classifier.cuda()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return {
            'y_pred': F.softmax(x, dim=1)[:, 1],
            'y_pred_loss': x
        }


class Model(BaseModel):

    def init_model(self):
        config = self.config
        gpu = not self.args.cpu
        self.network = Network(config, gpu)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adadelta(self.network.parameters(),
                                        eps=config['delta'], rho=config['momentum'])

    def train_metrics(self, y_true, y_pred):
        return {
            'accuracy': metrics.accuracy(y_true, y_pred)
        }

    def eval_metrics(self, y_true, y_pred):
        return {
            'accuracy': metrics.accuracy(y_true, y_pred),
            'auroc': metrics.auroc(y_true, y_pred)
        }

    def get_hyperband_reward(self, result):
        return -result['val_loss']
