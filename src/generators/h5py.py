from __future__ import print_function, absolute_import

from glob import glob
import h5py

import numpy as np

from .matrix import MatrixGen

class H5pyGen(MatrixGen):

    def __init__(self, glob_str, batch_size=None, shuffle=False, process_x_y=lambda X, Y: (X, Y)):
        files = [h5py.File(path) for path in glob(glob_str)]
        X, Y = zip(*[(file['data'][()], file['label'][()]) for file in files])
        X, Y = np.concatenate(X, axis=0), np.concatenate(Y, axis=0)
        X, Y = process_x_y(X, Y)
        super(H5pyGen, self).__init__(X, Y, batch_size, shuffle)

