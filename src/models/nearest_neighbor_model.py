from __future__ import print_function, absolute_import
from sklearn.neighbors import KNeighborsClassifier

class NearestNeighborModel(object):
    def __init__(self, config, args):
        self.config = config
        params = dict(n_neighbors=5, n_jobs=10)
        params.update(config.params)
        self.knn = KNeighborsClassifier(**params)

    def fit(self, train_gen, val_gen, stop_epoch, early_stopping=False):
        