import re

import pandas as pd
import torch

from util import numpy_to_builtin, load_json, save_json, save_text, Path

class Config(object):

    def __init__(self, res, params):
        self.res = Path(res).mk()
        self.params = self.load_params() if self.path.exists() else params
        self.name = self.res._real._name
        
        if not self.path.exists():
            self.save()

    @property
    def path(self):
        return self.res / 'config.json'
    
    @classmethod
    def from_path(cls, path):
        if Path(path).exists():
            args = load_json(path)
            return cls(args['result'], args['params'])
        else:
            return None
    
    def load_params(self):
        return load_json(self.path)['params']
    
    def save(self, force=False):
        if not force and self.path.exists():
            print('Not saving config %s, already exists and "force" is not specified' % self.path)
        else:
            save_json(self.path, dict(result=self.res, params=numpy_to_builtin(self.params)))


    @property
    def train_results(self):
        return self.res / 'train_results.csv'
    
    def load_train_results(self):
        if self.train_results.exists():
            return pd.read_csv(self.train_results, index_col=0)
        return None

    def save_train_results(self, results):
        results.to_csv(self.train_results, float_format='%.6g')


    @property
    def test_result(self):
        return self.res / 'test_result.json'

    def load_test_result(self):
        if self.test_result.exists():
            return load_json(self.test_result)
        return None

    def save_test_result(self, result):
        save_json(self.test_result, numpy_to_builtin(result))


    @property
    def best_reward(self):
        return self.res / 'best_reward.json'
    
    def save_best_reward(self, reward, epoch):
        save_json(self.best_reward, dict(reward=reward, epoch=epoch))
    
    def load_best_reward(self):
        if self.best_reward.exists():
            reward_dict = load_json(self.best_reward)
            return reward_dict['reward'], reward_dict['epoch']
        return None


    @property
    def stopped_early(self):
        return self.res / 'stopped_early'
    
    def set_stopped_early(self):
        save_text(self.stopped_early, '')

    
    @property
    def models(self):
        return (self.res / 'models').mk()
    
    def model_save(self, epoch):
        return self.models / ('model-%s.pth' % epoch)

    def get_saved_model_epochs(self):
        _, save_paths = self.models.ls()
        if len(save_paths) == 0:
            return []
        match_epoch = lambda path: re.match('.+/model-(\d+)\.pth', path)
        return [int(m.groups()[0]) for m in (match_epoch(p) for p in save_paths) if m is not None]

    def load_model_state(self, epoch=None, path=None):
        if epoch is not None:
            path = self.model_save(epoch)
        save_path = Path(path)
        if save_path.exists():
            return torch.load(save_path)
        return None

    def save_model_state(self, epoch, state):
        save_path = self.model_save(epoch)
        torch.save(state, save_path)
        return save_path

    def load_max_model_state(self, min_epoch=0):
        epochs = self.get_saved_model_epochs()
        if len(epochs) == 0:
            print('No saved model found in %s' % self.models)
            return None
        epoch = max(epochs)
        if epoch <= min_epoch:
            print('Model is already at epoch %s, no need to load' % min_epoch)
            return None
        return self.load_model_state(epoch=epoch)

    @property
    def model_best(self):
        return self.models / 'best_model.pth'

    def load_best_model_state(self):
        if self.model_best.exists():
            return self.load_model_state(path=self.model_best)
        print('No saved model found in %s' % self.models)
        return None

    def link_model_best(self, model_save):
        self.model_best.rm().link(Path(model_save).rel(self.models))

