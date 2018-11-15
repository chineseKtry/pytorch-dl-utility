from models.base_model import BaseModel
import metrics

class ClassificationModel(BaseModel):
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

    def get_hyperband_reward(self, result):
        return -result.loc['val_loss']
