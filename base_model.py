import os
import re
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
import metrics
import util


class BaseModel(object):

    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.device = 'cpu' if args.cpu else 'cuda'

        self.epoch = 0
        self.init_model()
        self.network.to(self.device)

    def init_model(self, network, criterion, optimizer, constraints=[]):
        self.network = network
        self.criterion = criterion
        self.optimizer = optimizer
        self.constraints = constraints
        if self.args.debug:
            print(self.network)
            print(self.criterion)
            print(self.optimizer)

    def fit(self, train_generator, val_generator, stop_epoch):
        self.load()
        if self.epoch >= stop_epoch:
            print('Already completed %s epochs' % self.epoch)
            return
        
        e_counter = util.progress_manager.counter(
            total=stop_epoch, desc='%s. Epoch' % self.config.name, unit='epoch', leave=False)
        e_counter.update(self.epoch)
        for epoch in xrange(self.epoch, stop_epoch):
            self.epoch = epoch + 1
            start = time()
            t_results = pd.DataFrame([self.fit_batch(xy) for xy in train_generator])
            epoch_result = t_results.mean(axis=0).add_prefix('train_')

            if val_generator:
                v_result = pd.Series(self.evaluate(val_generator)).add_prefix('val_')
                epoch_result = epoch_result.append(v_result)
            epoch_result['hyperband_reward'] = self.get_hyperband_reward(epoch_result)
            epoch_result['execution_time'] = '%.5g' % (time() - start)
            
            self.config.put_train_result(self.epoch, epoch_result)
            print('Epoch %s:\n%s\n' % (self.epoch, epoch_result.to_string(header=False)))

            e_counter.update()
        e_counter.close()
        self.save()
        self.config.save_train_results()

    def forward(self, xy):
        xy_t = map(lambda t: torch.from_numpy(t).to(self.device), xy)
        if len(xy_t) == 1:
            xy_t = xy_t + [None]
        x_t, y_t = xy_t
        pred = self.network(x_t)
        if type(pred) == dict:
            y_pred_t = pred['y_pred']
            y_pred_loss_t = pred['y_pred_loss']
        else:
            y_pred_t = y_pred_loss_t = pred
        y_pred = y_pred_t.cpu().detach().numpy()
        if y_t is None:
            return y_pred, None
        loss = self.criterion(y_pred_loss_t, y_t)
        return y_pred, loss

    def fit_batch(self, xy):
        self.network.train()
        y_pred, loss = self.forward(xy)

        for module_name, module in self.network.named_modules():
            for constraint in self.constraints:
                constraint.apply(module_name, module)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        (x, y) = xy
        results = self.train_metrics(y, y_pred)
        results['loss'] = loss.item()
        return results

    def predict(self, generator):
        self.network.eval()
        losses, y_preds = [], []
        with torch.no_grad():
            for xy in generator:
                y_pred_batch, loss = self.forward(xy)
                losses.append(loss and loss.item())
                y_preds.append(y_pred_batch)
        y_pred = np.concatenate(y_preds, axis=0)
        if generator.get_Y() is None:
            return y_pred
        return y_pred, np.mean(losses)

    def evaluate(self, generator):
        Y_pred, loss = self.predict(generator)
        Y = generator.get_Y()
        results = self.eval_metrics(Y, Y_pred)
        results['loss'] = loss
        return results

    def train_metrics(self, y_true, y_pred):
        raise NotImplementedError('Must override BaseModel.train_metrics')

    def eval_metrics(self, y_true, y_pred):
        raise NotImplementedError('Must override BaseModel.eval_metrics')

    def get_hyperband_reward(self, result):
        raise NotImplementedError('Must override BaseModel.get_hyperband_reward')

    def save(self):
        state = {
            'epoch': self.epoch,
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        return self.config.save_model_state(self.epoch, state)
        
    def load(self, epoch=None):
        if epoch is not None: # load a specific epoch
            state = self.config.load_model_state(epoch)
            assert state is not None, 'Epoch %s for model is not saved' % epoch
        else:
            state, epoch = self.config.load_best_model_state()
            if not state:
                return self
        self.network.load_state_dict(state['network'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.epoch = state['epoch']
        print('Loaded model from %s with %s iterations' % (self.config.model_dir, epoch))
        return self


class ConvClassificationNetwork(nn.Module):

    def __init__(self, features, classifier):
        super(ConvClassificationNetwork, self).__init__()
        self.features, self.classifier = features, classifier

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return {
            'y_pred': F.softmax(x, dim=1)[:, 1],
            'y_pred_loss': x
        }


class ClassificationModel(BaseModel):

    def init_model(self, network, optimizer, constraints=[]):
        super(ClassificationModel, self).init_model(network, nn.CrossEntropyLoss(), optimizer, constraints=constraints)
        
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
        return -result.loc['val_loss']


class ConvRegressionNetwork(nn.Module):

    def __init__(self, features, regressor):
        super(ConvRegressionNetwork, self).__init__()
        self.features, self.regressor = features, regressor

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x[:, 0]


class RegressionModel(BaseModel):

    def init_model(self, network, optimizer, constraints=[]):
        super(RegressionModel, self).init_model(network, nn.MSELoss(), optimizer, constraints=constraints)
        
    def train_metrics(self, y_true, y_pred):
        return {
            'mse': metrics.mse(y_true, y_pred)
        }

    def eval_metrics(self, y_true, y_pred):
        return {
            'mse': metrics.mse(y_true, y_pred),
            'pearson': metrics.pearson(y_true, y_pred)
        }

    def get_hyperband_reward(self, result):
        return -result.loc['val_loss']



