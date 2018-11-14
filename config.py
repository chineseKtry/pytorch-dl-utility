from glob import glob
import re
import os

import pandas as pd
import torch

import util

class Config:
    def __init__(self, result_dir, from_best=False, config_name='', config_dict={}):
        self.result_dir = result_dir
        
        self.best_dir = os.path.join(self.result_dir, 'best_config')
        if from_best:
            best_config_path = os.path.join(self.best_dir, 'config.json')
            if os.path.exists(best_config_path):
                config_dict = util.load_json(best_config_path)

        self.params = config_dict
        self.name = config_name
        if config_dict:
            self.name = ','.join(sorted('%s=%s' % (k, v) for k, v in config_dict.items()))
        if not self.name: # empty config
            return

        self.save_dir = os.path.join(self.result_dir, self.name)

        self.model_dir = os.path.join(self.save_dir, 'models')
        self.config_path = os.path.join(self.save_dir, 'config.json')
        self.train_results_path = os.path.join(self.save_dir, 'train_results.csv')
        self.test_result_path = os.path.join(self.save_dir, 'test_result.json')
        for d in [self.save_dir, self.model_dir]:
            util.makedirs(d)
        
        self.save_config()

        self.train_results = None
    

    # best config directory link
    def exist_best(self):
        return os.path.islink(self.best_dir)
    
    def link_as_best(self):
        if os.path.islink(self.best_dir):
            os.remove(self.best_dir)
        os.symlink(self.name, self.best_dir)
    

    # config file
    def save_config(self):
        util.save_json(self.params, self.config_path)
    

    # model files
    def save_model_state(self, epoch, state):
        save_path = os.path.join(self.model_dir, 'model-%s.pth' % epoch)
        torch.save(state, save_path)
        return save_path
    
    def load_model_state(self, epoch):
        save_path = os.path.join(self.model_dir, 'model-%s.pth' % epoch)
        if not os.path.exists(save_path):
            return None
        return torch.load(save_path)
    
    def _get_saved_model_epochs(self):
        save_paths = glob(os.path.join(self.model_dir, 'model-*.pth'))
        if len(save_paths) == 0:
            return []
        extract_epoch = lambda path: int(re.match('.+/model-(\d+)\.pth', path).groups()[0])
        return map(extract_epoch, save_paths)

    def load_max_model_state(self, curr_epoch=0):
        epochs = self._get_saved_model_epochs()
        if len(epochs) == 0:
            print('No saved model found in %s' % self.model_dir)
            return {}, 0
        epoch = max(epochs)
        if curr_epoch >= epoch:
            print('Model is already at epoch %s, no need to load' % curr_epoch)
            return {}, 0
        return self.load_model_state(epoch), epoch
    
    def load_best_model_state(self, metric='hyperband_reward'):
        self.load_train_results()
        epochs = self._get_saved_model_epochs()
        if len(epochs) == 0:
            print('No saved model found in %s' % self.model_dir)
            return {}, 0
        epoch = self.train_results.loc[epochs, metric].idxmax()
        return self.load_model_state(epoch), epoch


    # training results
    def load_train_results(self):
        if self.train_results is not None:
            return self.train_results
        elif os.path.exists(self.train_results_path):
            self.train_results = pd.read_csv(self.train_results_path, index_col=0)
            return self.train_results
        else:
            return None
    
    def save_train_results(self):
        self.train_results.to_csv(self.train_results_path, float_format='%.6g')
    
    def get_train_result(self, epoch):
        self.load_train_results()
        if self.train_results is not None and epoch in self.train_results.index:
            return self.train_results.loc[epoch]
        else:
            return None
    
    def put_train_result(self, epoch, epoch_result):
        self.load_train_results()
        if self.train_results is None:
            self.train_results = pd.DataFrame([epoch_result], index=pd.Series([epoch], name='epoch'))
        else:
            self.train_results.loc[epoch] = epoch_result


    # test results
    def save_test_result(self, result):
        util.save_json(result, self.test_result_path)

    def load_test_result(self):
        if not os.path.exists(self.test_result_path):
            return None
        return util.load_json(self.test_result_path)
