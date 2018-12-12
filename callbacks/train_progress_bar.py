from __future__ import print_function, absolute_import
from .callbacks import Callback

from util.progress_bar import RangeProgress

class TrainProgressBar(Callback):
    def on_train_start(self, model, train_state):
        self.prog = None
        if train_state.stop: return
        self.prog = iter(RangeProgress(model.epoch, train_state.stop_epoch, desc='%s. Epoch' % self.config.name))
    
    def on_epoch_end(self, model, train_state):
        next(self.prog)

    def on_train_end(self, model, train_state):
        if self.prog:
            self.prog.close()
