from __future__ import print_function, absolute_import

import numpy as np

class Model(object):

    def __init__(self, config, cpu=False, debug=False):
        self.config = config
        self.debug = debug
        self.device = 'cpu' if cpu else 'cuda'
    
        self.init_model()
    
    @classmethod
    def get_params(cls, config):
        return {}
    
    def init_model(self):
        raise NotImplementedError('Must implement init_model')
    
    def get_train_data(self):
        raise NotImplementedError('Must implement get_train_data if get_train_val_data not implemented')

    def get_val_data(self):
        return None
    
    def get_train_val_data(self):
        return self.get_train_data(), self.get_val_data()
    
    def get_test_data(self):
        raise NotImplementedError('Must implement get_test_data')

    def get_pred_data(self, pred_in_path):
        raise NotImplementedError('Must implement get_pred_data')
    
    def save_pred(self, pred_out_path, pred):
        np.save(pred_out_path, pred)
    
    def fit(self):
        raise NotImplementedError('Must implement fit')
    
    def evaluate(self, gen):
        raise NotImplementedError('Must implement evaluate')
    
    def predict(self):
        raise NotImplementedError('Must implement predict')

