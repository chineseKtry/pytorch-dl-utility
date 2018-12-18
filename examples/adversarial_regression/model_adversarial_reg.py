from __future__ import print_function

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from batch_generator import H5pyBatchGenerator

from base_model import RegressionModel, ConvRegressionNetwork
from constraints import MaxNormConstraint
from adversarial import AdversarialGenerator, AdversarialH5pyBatchGenerator

def get_config():
    return {
        'dropout': 0.3,
        'lr': 0.0001
    }

process_x_y = lambda X, Y: (X.astype(np.float32), Y.squeeze().astype(np.float32))
process_x_y_adv = lambda X, Y: (X.astype(np.float32), Y)
dna_process_x_y_adv = lambda X, Y: (X.astype(np.float32), np.concatenate([Y,1-Y],axis=1).astype(np.long))
dna_process_x_y = lambda X, Y: (X.astype(np.float32), (1-Y).squeeze().astype(np.long))

def get_train_generator(data_path, batch_size,seq='aa'):
    process = dna_process_x_y if seq == 'dna' else process_x_y
    data_str = 'train.h5.batch*' if seq == 'dna' else 'data.train.h5.batch*'
    return H5pyBatchGenerator(os.path.join(data_path, data_str), batch_size=batch_size, shuffle=True, process_x_y=process)


def get_adversarial_train_generator(data_path, batch_size, model, epsilon, seq='aa', criterion=nn.MSELoss()):
    adv_generator = AdversarialGenerator(model, criterion, epsilon=epsilon, single_aa=False, ohe_output=False, use_cuda=True)
    process = dna_process_x_y_adv if seq == 'dna' else process_x_y_adv
    data_str = 'train.h5.batch*' if seq == 'dna' else 'data.train.h5.batch*'    
    return AdversarialH5pyBatchGenerator(os.path.join(data_path, data_str), adv_generator, batch_size=batch_size, shuffle=True, process_x_y=process,task='regression')


def get_virtual_adversarial_train_generator(data_path, batch_size, model, epsilon=1.0, criterion=nn.KLDivLoss()):
    adv_generator = AdversarialGenerator(model, criterion ,ohe_output=False,use_cuda=True)
    process_x_y_adv = lambda X, Y : (X.astype(np.float32), None)
    return AdversarialH5pyBatchGenerator(os.path.join(data_path, 'data.train.h5.batch*'), adv_generator, batch_size=batch_size, shuffle=True, process_x_y=process_x_y_adv)


def get_val_generator(data_path, batch_size=None, seq='aa'):
    process = dna_process_x_y if seq == 'dna' else process_x_y
    data_str = 'valid.h5.batch*' if seq == 'dna' else 'data.valid.h5.batch*'
    return H5pyBatchGenerator(os.path.join(data_path, data_str), batch_size=batch_size, process_x_y=process)


def get_test_generator(data_path, batch_size=None, seq='aa'):
    process = dna_process_x_y if seq == 'dna' else process_x_y
    data_str = 'test.h5.batch*' if seq == 'dna' else 'data.test.h5.batch*'
    return H5pyBatchGenerator(os.path.join(data_path, data_str), batch_size=batch_size, process_x_y=process)

def get_adversarial_test_generator(data_path, batch_size, model, epsilon, seq='aa', criterion=nn.MSELoss()):
    adv_generator = AdversarialGenerator(model, criterion, epsilon=epsilon, single_aa=False, ohe_output=False, use_cuda=True)
    process = dna_process_x_y_adv if seq == 'dna' else process_x_y_adv
    data_str = 'test.h5.batch*' if seq == 'dna' else 'data.test.h5.batch*'    
    return AdversarialH5pyBatchGenerator(os.path.join(data_path, data_str), adv_generator, batch_size=batch_size, shuffle=True, process_x_y=process)


def get_model(config, save_dir, args):
    return Model(config, save_dir, args)

class Network_64x1_16(ConvRegressionNetwork):

    def __init__(self, config):
        features = nn.Sequential(
            nn.Conv2d(20, 64, (1, 5), padding=(0, 2)),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), stride=(1, 2))
        )
        classifier = nn.Sequential(
            nn.Linear(640, 16),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(16, 2)
        )
        super(Network_64x1_16, self).__init__(features, classifier)

class Model(RegressionModel):

    def init_model(self):
        config = self.config
        network_class = eval('Network_%s' % os.environ['network'])
        network = network_class(config.params)
        optimizer = optim.RMSprop(network.parameters(),
                                  lr=config.params['lr'], momentum=0.9, eps=1e-6)
        constraints = [MaxNormConstraint(3, '*')]
        super(Model, self).init_model(network, optimizer, constraints=constraints)

    def get_hyperband_reward(self, result):
        return result.loc['val_mse']