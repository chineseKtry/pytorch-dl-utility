from __future__ import print_function, absolute_import
from six.moves import range

import os
import re
from time import time

import numpy as np
import pandas as pd
import torch

from config import Config
from matchers import apply_matchers
import util


class BaseModel(object):

    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.device = 'cpu' if args.cpu else 'cuda'

        self.epoch = 0
        self.init_model()
        self.network.to(self.device)

    def init_model(self, network, optimizer, initializers=[], constraints=[]):
        self.network = network
        self.optimizer = optimizer
        self.constraints = constraints
        apply_matchers(self.network.named_modules(), initializers)
        if self.args.debug:
            print(self.network)
            print(self.optimizer)

    def fit(self, train_gen, val_gen, stop_epoch, early_stopping=False):
        config = self.config
        self.load()
        if self.epoch >= stop_epoch:
            print('Already completed %s epochs' % self.epoch)
            return
        
        self.network.train()
        
        e_counter = util.progress_manager.counter(
            total=stop_epoch, desc='%s. Epoch' % config.name, unit='epoch', leave=False)
        e_counter.update(self.epoch)
        stopped_early = False
        for epoch in range(self.epoch, stop_epoch):
            if early_stopping and self.epoch > 3:
                results = config.load_train_results().iloc[-4:]
                if results['hyperband_reward'].idxmax() == results.index[0]:
                    stopped_early = True
                    break

            self.epoch = epoch + 1
            start = time()
            t_results = pd.DataFrame([self.fit_batch(xy) for xy in train_gen])
            epoch_result = t_results.mean(axis=0).add_prefix('train_')

            if val_gen:
                v_result = pd.Series(self.evaluate(val_gen)).add_prefix('val_')
                epoch_result = epoch_result.append(v_result)
            epoch_result['hyperband_reward'] = self.get_hyperband_reward(epoch_result)
            epoch_result['execution_time'] = '%.5g' % (time() - start)
            
            config.put_train_result(self.epoch, epoch_result)
            print('Epoch %s:\n%s\n' % (self.epoch, epoch_result.to_string(header=False)))

            e_counter.update()
        e_counter.close()
        self.save()
        config.save_train_results()
        return self.epoch, epoch_result, stopped_early
    
    def to_torch(self, x):
        if type(x) == dict:
            return { k: self.to_torch(v) for k, v in x.items() }
        elif type(x) in [list, tuple]:
            return [self.to_torch(v) for v in x]
        return torch.from_numpy(x).to(self.device)
    
    def from_torch(self, t):
        if type(t) == dict:
            return { k: self.from_torch(v) for k, v in t.items() if v is not None }
        elif type(t) in [list, tuple]:
            return [self.from_torch(v) for v in t]
        
        x = t.detach().cpu().numpy()
        if x.size == 1 or np.isscalar(x):
            return np.asscalar(x)
        return x

    def fit_batch(self, xy):
        loss_t, pred_t = self.network(*self.to_torch(xy))

        self.optimizer.zero_grad()
        loss_t.backward()
        self.optimizer.step()
        
        apply_matchers(self.network.named_modules(), self.constraints)
        _, y = xy
        return self.train_metrics(y, self.from_torch(pred_t))
    
    def reduce_preds(self, preds):
        def op(preds_value):
            if np.isscalar(preds_value[0]): # take the means of scalar values
                return np.mean(preds_value)
            else:
                return np.concatenate(preds_value) # concatenate np arrays
        if type(preds[0]) == dict:
            return { key: op(value) for key, value in pd.DataFrame(preds).iteritems() }
        else:
            return op(preds)
    
    def evaluate(self, gen):
        self.network.eval()
        with torch.no_grad():
            preds = [self.from_torch(self.network(*self.to_torch(xy))[1]) for xy in gen]
        return self.eval_metrics(gen.get_Y(), self.reduce_preds(preds))

    def predict(self, gen):
        self.network.eval()
        with torch.no_grad():
            preds = [self.from_torch(self.network(self.to_torch(x), None)[1]) for x in gen]
        return self.reduce_preds(preds)

    def train_metrics(self, y_true, pred):
        raise NotImplementedError('Must override BaseModel.train_metrics')

    def eval_metrics(self, y_true, pred):
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
