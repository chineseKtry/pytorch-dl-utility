import glob
import h5py
import os

import numpy as np

class MatrixBatchGenerator(object):

    def __init__(self, X, Y=None, batch_size=None, shuffle=False):
        assert len(X) == len(Y), 'X and Y have mismatched lengths (%s and %s)' % (len(X), len(Y))
        self.N = len(X)
        if shuffle:
            indices = np.random.permutation(self.N)
            self.X, self.Y = X[indices], (None if Y is None else Y[indices])
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
            return (self.X[start: self.i],)
        else:
            return (self.X[start: self.i], self.Y[start: self.i])

    def get_X_shape(self):
        return self.X.shape[1:]

    def get_Y(self):
        return self.Y


class H5pyBatchGenerator(MatrixBatchGenerator):

    def __init__(self, data_dir, prefix, batch_size=None, shuffle=False, process_x_y=lambda X, Y: (X, Y)):
        paths = glob.glob(os.path.join(data_dir, prefix + '*.h5*'))
        files = [h5py.File(path) for path in paths]
        X, Y = zip(*[(file['data'][()], file['label'][()]) for file in files])
        X, Y = np.concatenate(X, axis=0), np.concatenate(Y, axis=0)
        X, Y = process_x_y(X, Y)
        super(H5pyBatchGenerator, self).__init__(X, Y, batch_size, shuffle)
