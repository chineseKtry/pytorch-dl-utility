from __future__ import print_function, absolute_import
from .callbacks import Callback

class ModelSaver(Callback):
    def __init__(self, config, min_save_period=5):
        super(ModelSaver, self).__init__(config)
        self.min_save_period = min_save_period

    def on_train_start(self, model, train_state):
        if train_state.stop: return
        model.set_state(self.config.load_max_model_state(min_epoch=model.epoch))
        self.last_save_epoch = model.epoch

    def on_epoch_end(self, model, train_state):
        if model.epoch - self.last_save_epoch >= self.min_save_period and train_state.get('save_epoch', False):
            save_path = self.config.save_model_state(model.epoch, model.get_state())
            self.config.link_model_best(save_path)
            self.last_save_epoch = model.epoch
            print('Saved model to %s' % save_path)

    def on_train_end(self, model, train_state):
        if model.epoch > 0:
            self.config.save_model_state(model.epoch, model.get_state())
