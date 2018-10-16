'imports and definitions shared by various defs files'

import numpy as np
import json

from math import log, sqrt
from time import time
from pprint import pprint

import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score as AUC, log_loss, accuracy_score as accuracy
from sklearn.metrics import mean_squared_error as MSE, mean_absolute_error as MAE

class Model:
    def __init__(self, network, criterion, optimizer):
        self.network = network
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch = 0

    def train(self, X, Y, epochs, batch_size): # TODO checkpointing and early stopping
        self.network.train()
        N = len(X)
        indices = np.arange(N)
        np.random.shuffle(indices)

        while self.epoch < epochs:
            for i in xrange(0, N, batch_size):
                x = torch.from_numpy(X[i: i + batch_size])
                y = torch.from_numpy(np.argmax(Y[i: i + batch_size], axis=1))

                y_pred = self.network(x.float())
                loss = self.criterion(y_pred, y.long())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.epoch += 1
            print('epoch', self.epoch, 'loss', loss.item())
        return self

    def evaluate(self, X, Y): # TODO generalize
        self.network.eval()
        y_pred_logit = self.network(torch.from_numpy(X).float())
        max_pred, y_pred = torch.max(y_pred_logit, 1)
        Y_val_value = torch.from_numpy(np.argmax(Y, axis=1)).long()
        loss = self.criterion(y_pred_logit, Y_val_value).item()
        accuracy = y_pred.eq(Y_val_value).float().mean()
        return loss, accuracy

    def predict(self, X): # TODO generalize
        self.network.eval()
        y_pred_logit = self.network(torch.from_numpy(X).float())
        y_pred = F.softmax(y_pred_logit, dim=1)
        return y_pred

    def save_checkpoint(self, checkpoint_base):
        state = {
            'epoch': self.epoch,
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, checkpoint_base + '.pth')
        # TODO save best and last
        return self

    def load_checkpoint(self, checkpoint_file):
        state = torch.load(checkpoint_file)
        self.network.load_state_dict(state['network'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.epoch = state['epoch']
        return self


# handle floats which should be integers
# works with flat params
def handle_integers(params):
    new_params = {}
    for k, v in params.items():
        if type(v) == float and int(v) == v:
            new_params[k] = int(v)
        else:
            new_params[k] = v
    return new_params


def load_json(path):
    with open(path, 'r+') as f:
        return json.load(f)


def save_json(dict_, path):
    with open(path, 'w+') as f:
        json.dump(dict_, f, indent=4, sort_keys=True)


def train_and_eval_sklearn_classifier(clf, data):

    x_train = data['x_train']
    y_train = data['y_train']

    x_test = data['x_test']
    y_test = data['y_test']

    clf.fit(x_train, y_train)

    try:
        p = clf.predict_proba(x_train)[:, 1]   # sklearn convention
    except IndexError:
        p = clf.predict_proba(x_train)

    ll = log_loss(y_train, p)
    auc = AUC(y_train, p)
    acc = accuracy(y_train, np.round(p))

    print '\n# training | log loss: {:.2%}, AUC: {:.2%}, accuracy: {:.2%}'.format(ll, auc, acc)

    try:
        p = clf.predict_proba(x_test)[:, 1]    # sklearn convention
    except IndexError:
        p = clf.predict_proba(x_test)

    ll = log_loss(y_test, p)
    auc = AUC(y_test, p)
    acc = accuracy(y_test, np.round(p))

    print '# testing  | log loss: {:.2%}, AUC: {:.2%}, accuracy: {:.2%}'.format(ll, auc, acc)

    # return { 'loss': 1 - auc, 'log_loss': ll, 'auc': auc }
    return {'loss': ll, 'log_loss': ll, 'auc': auc}


def train_and_eval_sklearn_regressor(reg, data):

    x_train = data['x_train']
    y_train = data['y_train']

    x_test = data['x_test']
    y_test = data['y_test']

    reg.fit(x_train, y_train)
    p = reg.predict(x_train)

    mse = MSE(y_train, p)
    rmse = sqrt(mse)
    mae = MAE(y_train, p)

    print '\n# training | RMSE: {:.4f}, MAE: {:.4f}'.format(rmse, mae)

    #

    p = reg.predict(x_test)

    mse = MSE(y_test, p)
    rmse = sqrt(mse)
    mae = MAE(y_test, p)

    print '# testing  | RMSE: {:.4f}, MAE: {:.4f}'.format(rmse, mae)

    return {'loss': rmse, 'rmse': rmse, 'mae': mae}
