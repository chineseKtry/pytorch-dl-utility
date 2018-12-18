from __future__ import print_function

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from batch_generator import H5pyBatchGenerator

from base_model import ClassificationModel, ConvClassificationNetwork
from constraints import MaxNormConstraint
from adversarial import AdversarialGenerator, AdversarialH5pyBatchGenerator

def get_config():
    return {
        'dropout': 0.3,
        'lr': 0.0001
    }

process_x_y = lambda X, Y: (X.astype(np.float32), Y[:, 1].astype(np.long))
process_x_y_adv = lambda X, Y: (X.astype(np.float32), Y.astype(np.long))
dna_process_x_y_adv = lambda X, Y: (X.astype(np.float32), np.concatenate([Y,1-Y],axis=1).astype(np.long))
dna_process_x_y = lambda X, Y: (X.astype(np.float32), (1-Y).squeeze().astype(np.long))

def get_train_generator(data_path, batch_size,seq='aa'):
    process = dna_process_x_y if seq == 'dna' else process_x_y
    data_str = 'train.h5.batch*' if seq == 'dna' else 'data.train.h5.batch*'
    return H5pyBatchGenerator(os.path.join(data_path, data_str), batch_size=batch_size, shuffle=True, process_x_y=process)


def get_adversarial_train_generator(data_path, batch_size, model, epsilon, seq='aa', criterion=nn.CrossEntropyLoss()):
    adv_generator = AdversarialGenerator(model, criterion, epsilon=epsilon, single_aa=False, ohe_output=False, use_cuda=True)
    process = dna_process_x_y_adv if seq == 'dna' else process_x_y_adv
    data_str = 'train.h5.batch*' if seq == 'dna' else 'data.train.h5.batch*'    
    return AdversarialH5pyBatchGenerator(os.path.join(data_path, data_str), adv_generator, batch_size=batch_size, shuffle=True, process_x_y=process)


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

def get_adversarial_test_generator(data_path, batch_size, model, epsilon, seq='aa', criterion=nn.CrossEntropyLoss()):
    adv_generator = AdversarialGenerator(model, criterion, epsilon=epsilon, single_aa=False, ohe_output=False, use_cuda=True)
    process = dna_process_x_y_adv if seq == 'dna' else process_x_y_adv
    data_str = 'test.h5.batch*' if seq == 'dna' else 'data.test.h5.batch*'    
    return AdversarialH5pyBatchGenerator(os.path.join(data_path, data_str), adv_generator, batch_size=batch_size, shuffle=True, process_x_y=process)


def get_model(config, save_dir, args):
    return Model(config, save_dir, args)


class Network_32x1_16(ConvClassificationNetwork):

    def __init__(self, config):
        features = nn.Sequential(
            nn.Conv2d(20, 32, (1, 5), padding=(0, 2)),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), stride=(1, 2))
        )
        classifier = nn.Sequential(
            nn.Linear(320, 16),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(16, 2)
        )
        super(Network_32x1_16, self).__init__(features, classifier)


class Network_64x1_16(ConvClassificationNetwork):

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


class Network_32x2_16(ConvClassificationNetwork):

    def __init__(self, config):
        features = nn.Sequential(
            nn.Conv2d(20, 32, (1, 5), padding=(0, 2)),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), stride=(1, 2)),
            nn.Conv2d(32, 64, (1, 5), padding=(0, 2)),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), stride=(1, 2))
        )
        classifier = nn.Sequential(
            nn.Linear(320, 16),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(16, 2)
        )
        super(Network_32x2_16, self).__init__(features, classifier)


class Network_32_32(ConvClassificationNetwork):

    def __init__(self, config):
        features = lambda x: x
        classifier = nn.Sequential(
            nn.Linear(400, 32),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(32, 2)
        )
        super(Network_32_32, self).__init__(features, classifier)


class Network_32x1_16_filt3(ConvClassificationNetwork):

    def __init__(self, config):
        features = nn.Sequential(
            nn.Conv2d(20, 32, (1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), stride=(1, 2))
        )
        classifier = nn.Sequential(
            nn.Linear(320, 16),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(16, 2)
        )
        super(Network_32x1_16_filt3, self).__init__(features, classifier)


class Network_emb_32x1_16(ConvClassificationNetwork):

    def __init__(self, config):
        features = nn.Sequential(
            nn.Conv2d(20, 8, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(8, 64, (1, 5), padding=(0, 2)),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), stride=(1, 2))
        )
        classifier = nn.Sequential(
            nn.Linear(640, 16),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(16, 2)
        )
        super(Network_emb_32x1_16, self).__init__(features, classifier)

class Model(ClassificationModel):

    def init_model(self):
        config = self.config
        network_class = eval('Network_%s' % os.environ['network'])
        network = network_class(config.params)
        optimizer = optim.RMSprop(network.parameters(),
                                  lr=config.params['lr'], momentum=0.9, eps=1e-6)
        constraints = [MaxNormConstraint(3, '*')]
        super(Model, self).init_model(network, optimizer, constraints=constraints)

    def get_hyperband_reward(self, result):
        return result.loc['val_auroc']

####################

# FOR DNA SEQUENCES
class Network_32x2_16_dna(ConvClassificationNetwork):

    def __init__(self, config):
        features = nn.Sequential(
            nn.Conv2d(4, 32, (1, 5), padding=(0, 2)),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), stride=(1, 2)),
            nn.Conv2d(32, 64, (1, 5), padding=(0, 2)),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), stride=(1, 2))
        )
        classifier = nn.Sequential(
            nn.Linear(1600, 16),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(16, 2)
        )
        super(Network_32x2_16_dna, self).__init__(features, classifier)

class Network_32x2_16_dna2(ConvClassificationNetwork):

    def __init__(self, config):
        features = nn.Sequential(
            nn.Conv2d(4, 32, (1, 5), padding=(0, 2)),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), stride=(1, 2)),
            nn.Conv2d(32, 64, (1, 5), padding=(0, 2)),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), stride=(1, 2))
        )
        classifier = nn.Sequential(
            nn.Linear(3200, 64),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(64, 2)
        )
        super(Network_32x2_16_dna2, self).__init__(features, classifier)


##########

class Network_32_32_VAT(Network_32_32):
    
    def loss_class(self):
        self.loss = nn.CrossEntropyLoss()
    def loss_virtual(self):
        self.loss = nn.KLDivLoss()

class ModelVAT(Model):
    '''things to fix...
        1) fit() function - epoch counter for adversarial examples, 
    '''
    # override fit() function from Model
    def fit(self, train_generator, val_generator, stop_epoch):
        self.load()
        if self.epoch >= stop_epoch:
            print('Already completed %s epochs' % self.epoch)
            return
        
        e_counter = util.progress_manager.counter(
            total=stop_epoch, desc='%s. Epoch' % self.config.name, unit='epoch', leave=False)
        e_counter.update(self.epoch)

        # assign appropriate loss function
        if train_generator.__class__.__name__ != 'AdversarialH5pyBatchGenerator':
            self.network.loss_class()
        else:
            self.network.loss_virtual()

        for epoch in xrange(self.epoch, stop_epoch):

            start = time()
            t_results = pd.DataFrame([self.fit_batch(xy) for xy in train_generator])
            epoch_result = t_results.mean(axis=0).add_prefix('train_')
            epoch_result['execution_time'] = '%.5g' % (time() - start)

            if train_generator.__class__.__name__ != 'AdversarialH5pyBatchGenerator':
                self.epoch = epoch + 1
                if val_generator:
                    v_result = pd.Series(self.evaluate(val_generator)).add_prefix('val_')
                    epoch_result = epoch_result.append(v_result)
                epoch_result['hyperband_reward'] = self.get_hyperband_reward(epoch_result)
                self.config.put_train_result(self.epoch, epoch_result)
            else:
                print('Virtual Adversarial Training')
            print('Epoch %s:\n%s\n' % (self.epoch, epoch_result.to_string(header=False)))

            e_counter.update()

        e_counter.close()
        if train_generator.__class__.__name__ != 'AdversarialH5pyBatchGenerator':
            self.save()
            self.config.save_train_results()

    # def fit_batch(self, xy):
    #     self.network.train()
    #     y_pred, loss = self.forward(xy)

    #     for module_name, module in self.network.named_modules():
    #         for constraint in self.constraints:
    #             constraint.apply(module_name, module)

    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    #     (x, y) = xy
    #     results = self.train_metrics(y, y_pred)
    #     results['loss'] = loss.item()
    #     return results

    # # override forward & fit_batch
    # def forward(self, xy):
    #     xy_t = map(lambda t: torch.from_numpy(t).to(self.device), xy)
    #     if len(xy_t) == 1:
    #         xy_t = xy_t + [None]
    #     x_t, y_t = xy_t
    #     pred = self.network(x_t)
    #     if type(pred) == dict:
    #         y_pred_t = pred['y_pred']
    #         y_pred_loss_t = pred['y_pred_loss']
    #     else:
    #         y_pred_t = y_pred_loss_t = pred
    #     y_pred = y_pred_t.cpu().detach().numpy()
    #     if y_t is None:
    #         return y_pred, None
    #     loss = self.criterion(y_pred_loss_t, y_t)
    #     return y_pred, loss
