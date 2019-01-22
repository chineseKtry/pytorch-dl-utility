from models.nn_model import NNModel
import metrics

class ClassificationModel(NNModel):
    def train_metrics(self, y_true, pred):
        return {
            'loss': pred['loss'],
            'accuracy': metrics.accuracy(y_true, pred['y'])
        }

    def eval_metrics(self, y_true, pred):
        return {
            'loss': pred['loss'],
            'accuracy': metrics.accuracy(y_true, pred['y']),
            'auroc': metrics.auroc(y_true, pred['y'])
        }

    def reward(self, result):
        return -result.loc['val_loss']
