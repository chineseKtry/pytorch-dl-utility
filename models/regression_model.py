from base_model import BaseModel
import metrics

class RegressionModel(BaseModel):

    def init_model(self, network, optimizer, constraints=[]):
        super(RegressionModel, self).init_model(network, nn.MSELoss(), optimizer, constraints=constraints)
        
    def train_metrics(self, y_true, y_pred):
        return {
            'mse': metrics.mse(y_true, y_pred)
        }

    def eval_metrics(self, y_true, y_pred):
        return {
            'mse': metrics.mse(y_true, y_pred),
            'pearson': metrics.pearson(y_true, y_pred)
        }

    def get_hyperband_reward(self, result):
        return -result.loc['val_loss']

