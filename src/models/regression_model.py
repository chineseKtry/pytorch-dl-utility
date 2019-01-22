from __future__ import print_function, absolute_import

from models.nn_model import NNModel
import metrics

class RegressionModel(NNModel):

    def train_metrics(self, y_true, pred):
        return {
            'loss': pred['loss'],
            'mse': metrics.mse(y_true, pred['y'])
        }

    def eval_metrics(self, y_true, pred):
        return {
            'loss': pred['loss'],
            'mse': metrics.mse(y_true, pred['y']),
            'pearson': metrics.pearson(y_true, pred['y'])
        }

    def get_hyperband_reward(self, result):
        return -result.loc['val_loss']

