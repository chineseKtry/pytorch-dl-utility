from __future__ import print_function, absolute_import
from future.utils import implements_iterator

import os
import numpy as np

@implements_iterator
class MatrixGen(object):

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
        train_gen = MatrixGen(X[:num_train], Y[:num_train], batch_size)
        val_gen = MatrixGen(X[num_train:], Y[num_train:])
        return train_gen, val_gen
