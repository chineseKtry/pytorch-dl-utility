from future.utils import implements_iterator

from glob import glob
import h5py
import os

import numpy as np

import torch
from torch.utils import data

@implements_iterator
class MatrixBatchGenerator(object):

    def __init__(self, X, Y=None, batch_size=None, shuffle=False):
        assert Y is None or len(X) == len(Y), 'X and Y have mismatched lengths (%s and %s)' % (len(X), len(Y))
        self.N = len(X)
        if shuffle:
            indices = np.random.RandomState(seed=42).permutation(self.N)
            self.X, self.Y = X[indices], (None if Y is None else Y[indices])
        else:
            self.X, self.Y = X, Y
        self.batch_size = batch_size or self.N

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.N:
            raise StopIteration
        start = self.i
        self.i += self.batch_size
        if self.Y is None:
            return (self.X[start: self.i], None)
        else:
            return (self.X[start: self.i], self.Y[start: self.i])
    
    def get_X(self):
        return self.X

    def get_Y(self):
        return self.Y
    
    @classmethod
    def get_train_val_generator(cls, X, Y, val_ratio, batch_size):
        N = len(X)
        indices = np.random.permutation(N)
        X, Y = X[indices], Y[indices]
        num_train = int(N * val_ratio)
        train_gen = MatrixBatchGenerator(X[:num_train], Y[:num_train], batch_size)
        val_gen = MatrixBatchGenerator(X[num_train:], Y[num_train:])
        return train_gen, val_gen

class H5pyBatchGenerator(MatrixBatchGenerator):

    def __init__(self, glob_str, batch_size=None, shuffle=False, process_x_y=lambda X, Y: (X, Y)):
        files = [h5py.File(path) for path in glob(glob_str)]
        X, Y = zip(*[(file['data'][()], file['label'][()]) for file in files])
        X, Y = np.concatenate(X, axis=0), np.concatenate(Y, axis=0)
        X, Y = process_x_y(X, Y)
        super(H5pyBatchGenerator, self).__init__(X, Y, batch_size, shuffle)

