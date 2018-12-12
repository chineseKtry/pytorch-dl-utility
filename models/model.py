class Model(object):

    def __init__(self, exp, config, cpu=False, debug=False):
        self.exp = exp
        self.config = config
        self.params = config.params
        self.debug = debug
        self.device = 'cpu' if cpu else 'cuda'
    
        self.init_model()
    
    @classmethod
    def get_params(cls, exp):
        return {}
    
    def init_model(self):
        raise NotImplementedError('Must implement init_model')
    
    def get_train_data(self):
        raise NotImplementedError('Must implement get_train_data if get_train_val_data not implemented')

    def get_val_data(self):
        return None
    
    def get_train_val_data(self):
        return self.get_train_data(), self.get_val_data()
    
    def get_test_data(self):
        raise NotImplementedError('Must implement get_test_data')

    def get_pred(self, pred_key):
        return self.get_pred_data(pred_key), self.get_pred_saver(pred_key)
        
    def get_pred_data(self, pred_key):
        raise NotImplementedError('Must implement get_pred_data')
    
    def get_pred_saver(self, pred_key):
        raise NotImplementedError('Must implement get_pred_saver')
    
    def fit(self):
        raise NotImplementedError('Must implement fit')
    
    def test(self):
        raise NotImplementedError('Must implement test')

    def predict(self):
        raise NotImplementedError('Must implement predict')

    def save_pred(self, pred_key, pred_out):
        raise NotImplementedError('Must implement save_pred')

