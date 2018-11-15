from __future__ import print_function

import os

import numpy as np
import torch.nn as nn
import torch.optim as optim

from models.classification_model import ClassificationModel
from networks.conv_classification_network import ConvClassificationNetwork
from batch_generator import H5pyBatchGenerator
from constraints import MaxNormConstraint


def get_config():
    return {
        'dropout': np.random.choice([0.1, 0.5, 0.75]),
        'delta': np.random.choice([1e-4, 1e-6, 1e-8]),
        'momentum': np.random.choice([0.9, 0.99, 0.999])
    }

process_x_y = lambda X, Y: (X.astype(np.float32), Y[:, 1].astype(np.long))


def get_train_generator(data_dir, batch_size):
    return H5pyBatchGenerator(os.path.join(data_dir, 'train.h5.batch*'), batch_size=batch_size,
                              shuffle=True, process_x_y=process_x_y)


def get_val_generator(data_dir):
    return H5pyBatchGenerator(os.path.join(data_dir, 'valid.h5.batch*'), process_x_y=process_x_y)


def get_test_generator(data_dir):
    return H5pyBatchGenerator(os.path.join(data_dir, 'test.h5.batch*'), process_x_y=process_x_y)


def get_pred_generator(pred_path):
    process_x_y_pred = lambda X, Y: (X.astype(np.float32), None)
    return H5pyBatchGenerator(pred_path, process_x_y=process_x_y_pred)


class Network(ConvClassificationNetwork):

    def __init__(self, params):
        features = nn.Sequential(
            nn.Conv2d(4, 128, (1, 24), padding=(0, 11)),
            nn.ReLU(),
            nn.MaxPool2d((1, 100))
        )
        classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(32, 2)
        )
        super(Network, self).__init__(features, classifier)


class Model(ClassificationModel):

    def init_model(self):
        params = self.config.params
        network = Network(params)
        optimizer = optim.Adadelta(network.parameters(),
                                   eps=params['delta'], rho=params['momentum'])
        constraints = [MaxNormConstraint(3, '*')]
        super(Model, self).init_model(network, optimizer, constraints=constraints)
