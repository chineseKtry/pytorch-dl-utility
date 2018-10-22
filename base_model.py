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
        self.save_dir = save_dir
        self.model_dir = os.path.join(save_dir, 'models')
        self.result_dir = os.path.join(save_dir, 'results')
        for d in [self.save_dir, self.model_dir, self.result_dir]:
            util.makedirs(d)

        self.epoch = 0
        self.save_config()
        self.init_model()

    def init_model(self):
        raise NotImplementedError('Must override BaseModel.init_model')

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
            epoch_result = t_results.mean(axis=0).add_prefix('train_').to_dict()

            if val_generator:
                v_result = pd.Series(self.evaluate(val_generator)).add_prefix('val_').to_dict()
                epoch_result.update(v_result)
            epoch_result['hyperband_reward'] = self.get_hyperband_reward(epoch_result)
            epoch_result['execution_time'] = '%.5g' % (time() - start)
            print('Epoch %s: %s\n' % (self.epoch, util.format_json(epoch_result)))

            epoch_result['epoch'] = self.epoch
            self.save_result(epoch_result)
            e_counter.update()
        e_counter.close()

    def forward(self, xy_t):
        if len(xy_t) == 1:
            xy_t = xy_t + (None,)
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

    def forward_numpy(self, xy):
        xy_t = map(torch.from_numpy, xy)
        if not self.args.cpu:
            xy_t = map(lambda t: t.cuda(), xy_t)
        return self.forward(xy_t)

    def fit_batch(self, xy):
        self.network.train()
        y_pred, loss = self.forward_numpy(xy)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        (x, y) = xy
        results = self.train_metrics(y, y_pred)
        results['loss'] = loss.item()
        return results

    def predict(self, generator):
        self.network.eval()
        gpu = not self.args.cpu
        losses, y_preds = [], []
        for xy in generator:
            y_pred_batch, loss = self.forward_numpy(xy)
            losses.append(loss.item())
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

    def save_result(self, result):  # TODO maybe change to csv or database format
        util.save_json(result, os.path.join(self.result_dir, 'train-%s.json' % self.epoch))

    def load_result(self, epoch):
        result_path = os.path.join(self.result_dir, 'train-%s.json' % epoch)
        if os.path.exists(result_path):
            print('Loaded result for epoch %s at %s' % (epoch, result_path))
            return util.load_json(result_path)
        else:
            print('Could not find result for epoch %s at %s' % (epoch, result_path))
            return None

    def save_config(self):
        util.save_json(self.config, os.path.join(self.save_dir, 'config.json'))


class ConvClassificationNetwork(nn.Module):

    def __init__(self, features, classifier, gpu):
        super(ConvClassificationNetwork, self).__init__()
        if gpu:
            features, classifier = features.cuda(), classifier.cuda()
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

    def init_model(self, network, optimizer):
        config = self.config
        gpu = not self.args.cpu
        self.criterion = nn.CrossEntropyLoss()
        self.network = network
        self.optimizer = optimizer

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
        return -result['val_loss']
