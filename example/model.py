import glob
import h5py
import os
from pprint import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


config = {
    'dropout': np.random.choice([0.1, 0.5, 0.75]),  # TODO change to generator
    'delta': np.random.choice([1e-4, 1e-6, 1e-8]),
    'momentum': np.random.choice([0.9, 0.99, 0.999])
}


class Network(nn.Module):

    def __init__(self, config, input_shape):
        super(Network, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 128, (1, 24), padding=(0, 11)),
            nn.ReLU(),
            nn.MaxPool2d((1, 100))
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def accuracy(y_preds, y):
    return np.mean(np.argmax(y_preds, axis=1) == y)


class Model(Trainable):

    def _setup(self, config):
        process_x_y = lambda X, Y: (X.astype(np.float32), Y[:, 1].astype(np.long))
        c = config
        self.train_data = H5pyBatchGenerator(c['data'], 'train',
                                             batch_size=c['batch_size'], shuffle=True, process_x_y=process_x_y)
        self.val_data = H5pyBatchGenerator(c['data'], 'valid', process_x_y=process_x_y)
        self.epoch = 0
        self.network = Network(c, self.train_data.get_X_shape())
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adadelta(self.network.parameters(), eps=c['delta'], rho=c['momentum'])
        self.metrics = [accuracy] # TODO

    def feed_batch(self, x, y, train=False, metrics=[]):
        y_preds = self.network(torch.from_numpy(x))
        loss = self.criterion(y_preds, torch.from_numpy(y))
        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        y_preds = y_preds.detach().numpy()
        return [loss.item()] + [metric(y_preds, y) for metric in metrics]

    def _train(self):
        self.epoch += 1
        train_loss, train_accuracy = np.mean(
            [self.feed_batch(x, y, train=True, metrics=self.metrics) for x, y in self.train_data],
            axis=0)
        val_loss, val_accuracy = np.mean(
            [self.feed_batch(x, y, metrics=self.metrics) for x, y in self.val_data],
            axis=0)
        return {
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'reward': -val_loss
        }

    def _save(self, path):
        state = {
            'epoch': self.epoch,
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, os.path.join(path, 'model.pth'))
        return path

    def _restore(self, path):
        state = torch.load(os.path.join(path, 'model.pth'))
        self.network.load_state_dict(state['network'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.epoch = state['epoch']


class MatrixBatchGenerator(object):

    def __init__(self, X, Y=None, batch_size=None, shuffle=False):
        assert len(X) == len(Y), 'X and Y have mismatched lengths (%s and %s)' % (len(X), len(Y))
        self.N = len(X)
        if shuffle:
            indices = np.random.permutation(self.N)
            self.X, self.Y = X[indices], Y[indices]
        else:
            self.X, self.Y = X, Y
        self.batch_size = batch_size or self.N

    def __iter__(self):
        self.i = 0
        return self

    def next(self):
        if self.i >= self.N:
            raise StopIteration
        start = self.i
        self.i += self.batch_size
        if self.Y is None:
            return self.X[start: self.i]
        else:
            return self.X[start: self.i], self.Y[start: self.i]

    def get_X_shape(self):
        return self.X.shape[1:]


class H5pyBatchGenerator(MatrixBatchGenerator):

    def __init__(self, data_dir, prefix, batch_size=None, shuffle=False, process_x_y=lambda X, Y: (X, Y)):
        paths = glob.glob(os.path.join(data_dir, prefix + '*.h5*'))
        files = [h5py.File(path) for path in paths]
        X, Y = zip(*[(file['data'][()], file['label'][()]) for file in files])
        X, Y = np.concatenate(X, axis=0), np.concatenate(Y, axis=0)
        X, Y = process_x_y(X, Y)
        super(H5pyBatchGenerator, self).__init__(X, Y, batch_size, shuffle)

