class Callback(object):
    def __init__(self, config):
        self.config = config

    def on_train_start(self, model):
        pass

    def on_epoch_end(self, model, epoch, epoch_result):
        pass

    def on_train_end(self, model):
        pass
