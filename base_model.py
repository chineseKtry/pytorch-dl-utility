from glob import glob
import os
import re
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import metrics
import util


class BaseModel(object):

    def __init__(self, config, save_dir, args):
        self.config = config
        self.args = args
        self.device = 'cpu' if args.cpu else 'cuda'

        self.save_dir = save_dir
        self.model_dir = os.path.join(save_dir, 'models')
        self.config_path = os.path.join(save_dir, 'config.json')
        self.train_results_path = os.path.join(save_dir, 'train_results.csv')
        self.test_result_path = os.path.join(save_dir, 'test_result.json')
        for d in [self.save_dir, self.model_dir]:
            util.makedirs(d)

        self.epoch = 0
        self.save_config()
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
        if self.epoch >= stop_epoch:
            print('Already completed %s epochs' % self.epoch)
            return

        e_counter = util.progress_manager.counter(
            total=stop_epoch, desc='%s. Epoch' % util.get_config_name(self.config), unit='epoch', leave=False)
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
            
            self.append_train_result(self.epoch, epoch_result)
            print('Epoch %s:\n%s\n' % (self.epoch, epoch_result.to_string(header=False)))

            e_counter.update()
        e_counter.close()

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
        save_path = os.path.join(self.model_dir, 'model-%s.pth' % self.epoch)
        torch.save(state, save_path)
        return save_path

    def load(self):
        save_paths = glob(os.path.join(self.model_dir, 'model-*.pth'))
        if len(save_paths) == 0:
            print('No saved model found in %s' % self.model_dir)
            return
        extract_epoch = lambda path: int(re.match('.+/model-(\d+)\.pth', path).groups()[0])
        save_path = max(save_paths, key=extract_epoch)
        epoch = extract_epoch(save_path)
        if epoch == self.epoch:
            print('Model already at epoch %s, no need to load' % self.epoch)
            return
        state = torch.load(save_path)
        self.network.load_state_dict(state['network'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.epoch = state['epoch']
        print('Loaded model from %s with %s iterations' % (save_path, epoch))

    def load_train_results(self):
        if os.path.exists(self.train_results_path):
            self.train_results = pd.read_csv(self.train_results_path, index_col=0)
        else:
            self.train_results = None
        return self.train_results
    
    def save_train_results(self):
        if self.train_results is not None:
            self.train_results.to_csv(self.train_results_path, float_format='%.6g')
    
    def get_train_result(self, epoch):
        if self.train_results is not None and epoch in self.train_results.index:
            return self.train_results.loc[epoch]
        else:
            return None
    
    def append_train_result(self, epoch, epoch_result):
        if self.train_results is None:
            self.train_results = pd.DataFrame([epoch_result], index=pd.Series([epoch], name='epoch'))
        else:
            self.train_results.loc[epoch] = epoch_result

    def save_test_result(self, result):
        util.save_json(result, self.test_result_path)

    def save_config(self):
        util.save_json(self.config, self.config_path)


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


class ConvClassificationModel(BaseModel):

    def init_model(self, network, optimizer, constraints=[]):
        super(ConvClassificationModel, self).init_model(network, nn.CrossEntropyLoss(), optimizer, constraints=constraints)
        
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


class ConvRegressionModel(BaseModel):

    def init_model(self, network, optimizer, constraints=[]):
        super(ConvRegressionModel, self).init_model(network, nn.MSELoss(), optimizer, constraints=constraints)
        
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



