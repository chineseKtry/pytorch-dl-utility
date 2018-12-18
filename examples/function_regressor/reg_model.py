from __future__ import print_function, absolute_import

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from batch_generator import MatrixBatchGenerator
from models.regression_model import RegressionModel
from networks.conv_regression_network import ConvRegressionNetwork
from constraints import MaxNormConstraint
import metrics


def get_config():
    return {
        'dropout': np.random.choice([0.3, 0.5]),
        'lr': np.random.choice([1e-1, 1e-2, 1e-3, 1e-4, 1e-5]),
        'momentum': np.random.choice([0.5, 0.9])
    }

def get_train_generator(data_path, batch_size):
    X = pd.read_csv(os.path.join(data_path, 'X_train.csv'), index_col=0).values.astype(np.float32)
    Y = pd.read_csv(os.path.join(data_path, 'Y_train.csv'), index_col=0).values.astype(np.float32)
    return MatrixBatchGenerator.get_train_val_generator(X, Y[:, 0], 0.1, 100)


def get_test_generator(data_path):
    X = pd.read_csv(os.path.join(data_path, 'X_test.csv'), index_col=0).values.astype(np.float32)
    Y = pd.read_csv(os.path.join(data_path, 'Y_test.csv'), index_col=0).values.astype(np.float32)
    return MatrixBatchGenerator(X, Y[:, 0])


def get_pred_generator(pred_path):
    X = pd.read_csv(pred_path, index_col=0).values.astype(np.float32)
    return MatrixBatchGenerator(X)


class Network(ConvRegressionNetwork):

    def __init__(self, params):
        features = lambda x: x
        regressor = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(20, 1)
        )
        super(Network, self).__init__(features, regressor)
    
    def forward(self, x_t, y_t):
        loss_t, output_dict = super(Network, self).forward(x_t, y_t)
        output_dict['x'] = x_t
        output_dict['y_true'] = y_t
        return loss_t, output_dict

class Model(RegressionModel):

    def init_model(self):
        params = self.config.params
        network = Network(params)
        optimizer = optim.RMSprop(network.parameters(),
                                  lr=params['lr'], momentum=params['momentum'], eps=1e-6)
        constraints = [MaxNormConstraint(10, '*')]
        super(Model, self).init_model(network, optimizer, constraints=constraints)
    
    def train_metrics(self, y_true, pred):
        return self.eval_metrics(y_true, pred)

    def eval_metrics(self, y_true, pred):
        stats = super(Model, self).eval_metrics(y_true, pred)
        stats['mean_x'] = np.mean(pred['x'])
        stats['std_x'] = np.std(pred['x'])
        return stats

    def get_hyperband_reward(self, result):
        return -result.loc['val_mse']
