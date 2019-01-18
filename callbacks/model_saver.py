from __future__ import print_function, absolute_import
from .callbacks import Callback

class ModelSaver(Callback):
    def __init__(self, config):
        super(ModelSaver, self).__init__(config)

    def on_train_start(self, model, train_state):
        self.recorded_model = None
        if train_state.stop: return
        model.set_state(self.config.load_max_model_state(min_epoch=model.epoch))

    def on_epoch_end(self, model, train_state):
        if train_state.get('record_epoch', False):
            self.recorded_model = (model.epoch, model.get_state())
        if train_state.get('save_recorded_to_disk', False):
            self.save_recorded()

    def on_train_end(self, model, train_state):
        if self.recorded_model is not None:
            self.save_recorded()
        if model.epoch > 0 and not self.config.model_save(model.epoch).exists():
            save_path = self.config.save_model_state(model.epoch, model.get_state())
            print('Saved model at epoch %s to %s' % (model.epoch, save_path))
    
    def save_recorded(self):
        save_path = self.config.save_model_state(*self.recorded_model)
        self.config.link_model_best(save_path)
        print('Linked %s to new saved model %s' % (self.config.model_best, save_path))
        self.recorded_model = None