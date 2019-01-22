from __future__ import print_function, absolute_import

import pandas as pd

from . import Callback

class ResultMonitor(Callback):
    def __init__(self, config):
        super(ResultMonitor, self).__init__(config)
        self.early_stop = config.early_stop or False
        self.patience = config.early_stop
        
        self.best_reward = -float('inf')
        self.best_epoch = None
        if config.best_reward.exists():
            self.best_reward, self.best_epoch = config.load_best_reward()
        self.stopped_early = False

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

        reward_fn = model.reward

        reward = reward_fn(train_state.epoch_result)
        train_state.record_epoch = False
        train_state.save_recorded_to_disk = False
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_epoch = model.epoch
            train_state.record_epoch = True
            print('New best epoch %s with reward %s' % (self.best_epoch, self.best_reward))
        elif self.best_epoch == model.epoch - 1:
            train_state.save_recorded_to_disk = True
            self.config.save_best_reward(self.best_reward, self.best_epoch)

        if self.early_stop and model.epoch > self.patience:
            recent = self.train_results.iloc[-(self.patience + 1):]
            last_rewards = recent.apply(reward_fn, axis=1)
            if last_rewards.idxmax() == recent.index[0]:
                self.stopped_early = model.epoch
                train_state.stop = True
                self.config.set_stopped_early()
                print('Stopped early after %s / %s iterations' % (model.epoch, train_state.stop_epoch))
                return

    def on_train_end(self, model, train_state):
        if self.train_results is not None:
            self.config.save_train_results(self.train_results)
        if self.best_epoch == model.epoch:
            self.config.save_best_reward(self.best_reward, self.best_epoch)
    
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
