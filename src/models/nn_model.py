from __future__ import print_function, absolute_import
from six.moves import range

import os
import re
from time import time

import numpy as np
import pandas as pd
import torch

from ..util import Namespace, format_json
from ..util.torch import to_torch, from_torch
from . import Model
from ..matchers import apply_matchers
from ..callbacks.result_monitor import ResultMonitor
from ..callbacks.model_saver import ModelSaver
from ..callbacks.train_progress import TrainProgress

class NNModel(Model):

    def set_network(self, network):
        self.epoch = 0
        self.network = network.to(self.device)
        self.optimizer = network.optimizer
        apply_matchers(self.network.named_modules(), getattr(network, 'initializers', []))
        self.regularizers = getattr(network, 'regularizers', [])
        self.constraints = getattr(network, 'constraints', [])
        if self.debug:
            print(self.network)
    
    def fit(self, stop_epoch):
        callbacks = [ResultMonitor, ModelSaver, TrainProgress]
        callbacks = [cb(self.config) for cb in callbacks]

        self.network.train()
        train_gen, val_gen = self.get_train_val_data()

        train_state = Namespace(stop=False, stop_epoch=stop_epoch)
        [cb.on_train_start(self, train_state) for cb in callbacks]

        while not train_state.stop and self.epoch < stop_epoch:
            start = time()
            t_results = pd.DataFrame([self.fit_batch(xy) for xy in train_gen])

            self.epoch += 1
            epoch_result = t_results.mean(axis=0).add_prefix('train_')

            if val_gen:
                v_result = pd.Series(self.evaluate(val_gen)).add_prefix('val_')
                epoch_result = epoch_result.append(v_result)
            epoch_result['execution_time'] = '%.5g' % (time() - start)
            
            train_state.update(epoch_result=epoch_result)
            [cb.on_epoch_end(self, train_state) for cb in callbacks]

        [cb.on_train_end(self, train_state) for cb in callbacks]
        return self
    
    def fit_batch(self, xy):
        loss_t, pred_t = self.network(*to_torch(xy, device=self.device))

        loss_t = apply_matchers(self.network.named_modules(), self.regularizers, loss_t=loss_t)

        self.optimizer.zero_grad()
        loss_t.backward()
        self.optimizer.step()
        
        apply_matchers(self.network.named_modules(), self.constraints)
        _, y = xy
        return self.train_metrics(y, from_torch(pred_t))
    
    def predict(self, gen):
        self.network.eval()
        with torch.no_grad():
            preds = [from_torch(self.network(*to_torch(xy, device=self.device))[1]) for xy in gen]
        return reduce_preds(preds)

    def evaluate(self, gen):
        return self.eval_metrics(gen.get_Y(), self.predict(gen))
    
    def train_metrics(self, y_true, pred):
        raise NotImplementedError('Must override train_metrics')

    def eval_metrics(self, y_true, pred):
        raise NotImplementedError('Must override eval_metrics')

    def reward(self, epoch_result):
        return -epoch_result['val_loss']

    def get_state(self):
        return dict(epoch=self.epoch, network=self.network.state_dict(), optimizer=self.optimizer.state_dict())

    def set_state(self, state):
        if state is None: return self
        self.network.load_state_dict(state['network'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.epoch = state['epoch']
        return self

def reduce_preds(preds):
    def op(preds_value):
        if np.isscalar(preds_value[0]): # take the means of scalar values
            return np.mean(preds_value)
        else:
            return np.concatenate(preds_value) # concatenate np arrays
    if type(preds[0]) == dict:
        return { key: op(value) for key, value in pd.DataFrame(preds).iteritems() }
    else:
        return op(preds)
