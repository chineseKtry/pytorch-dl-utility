from __future__ import print_function, absolute_import

import pandas as pd

from .callbacks import Callback

class ResultMonitor(Callback):
    def on_train_start(self, model, train_state):
        self.train_results = None
        if train_state.stop: return
        if not train_state.stop and self.config.stopped_early.exists():
            train_state.stop = True
            print('Preempting training because already stopped early')
            return
        self.train_results = self.config.load_train_results()
        if self.train_results is not None and self.train_results.index[-1] >= train_state.stop_epoch:
            train_state.stop = True
            print('Preempting training because already trained past %s epochs' % train_state.stop_epoch)
            return
    
    def on_epoch_end(self, model, train_state):
        self.put_train_result(model.epoch, train_state.epoch_result)
        print('Epoch %s:\n%s\n' % (model.epoch, train_state.epoch_result.to_string(header=False)))

    def on_train_end(self, model, train_state):
        if self.train_results is not None:
            self.config.save_train_results(self.train_results)
    
    def put_train_result(self, epoch, result):
        if self.train_results is None:
            self.train_results = pd.DataFrame([result], index=pd.Series([epoch], name='epoch'))
        else:
            self.train_results.loc[epoch] = result

    def get_train_result(self, epoch):
        if self.train_results is None:
            return None
        else:
            return self.train_results.loc[epoch]
