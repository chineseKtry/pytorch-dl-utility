from __future__ import print_function

import numpy as np
import torch.nn as nn
import torch.optim as optim

from base_model import ConvClassificationModel, ConvClassificationNetwork
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


class Network(ConvClassificationNetwork):

    def __init__(self, config, gpu):
        features = nn.Sequential(
            nn.Conv2d(4, 128, (1, 24), padding=(0, 11)),
            nn.ReLU(),
            nn.MaxPool2d((1, 100))
        )
        classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(32, 2)
        )
        super(Network, self).__init__(features, classifier, gpu)


class Model(ConvClassificationModel):

    def init_model(self):
        config = self.config
        gpu = not self.args.cpu
        network = Network(config, gpu)
        optimizer = optim.Adadelta(network.parameters(),
                                   eps=config['delta'], rho=config['momentum'])
        super(Model, self).init_model(network, optimizer)
