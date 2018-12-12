from __future__ import print_function, absolute_import
from six.moves import range

import math

from util.progress_bar import ListProgress, RangeProgress
from callbacks.result_monitor import ResultMonitor
from callbacks.model_saver import ModelSaver

class Hyperband:

    def __init__(self, Model, exp, args):
        self.Model = Model
        self.exp = exp
        self.args = args
        
        self.max_iter = exp.hyper_epoch  # maximum iterations per configuration
        self.eta = math.e  # defines configuration downsampling rate

        logeta = lambda x: math.log(x) / math.log(self.eta)
        self.s_max = int(logeta(self.max_iter))
        self.B = (self.s_max + 1) * self.max_iter

    def run(self):
        Model = self.Model
        args = self.args
        exp = self.exp

        best_config = exp.config_best()
        if best_config is not None:
            print('Best config %s => %s already exists, terminating Hyperband' % (exp.best, best_config.name))
            return
        
        state = exp.load_hyperband_state()
        if state:
            print('Loaded Hyperband initial parameters to search from %s' % exp.hyperband_state)
        else:
            state = {}
            for s in range(self.s_max, -1, -1):
                # initial number of configurations
                n = int(math.ceil(self.B / self.max_iter / (s + 1) * self.eta ** s))

                # n random configurations
                state[s] = [Model.get_params(exp) for _ in range(n)]
            exp.save_hyperband_state(state)
            print('Generated Hyperband initial parameters to search to %s' % exp.hyperband_state)

        best_info = dict(reward=-float('inf'))
        for s in RangeProgress(self.s_max, -1, step=-1, desc='Sweeping s'):
            T = [exp.config(params=params) for params in state[s]]
            n = len(T)

            # initial number of iterations per config
            r = self.max_iter * self.eta ** (-s)

            for i in RangeProgress(0, s + 1, desc='s = %s. Sweeping i' % s):
                # Run each of the n configs for <iterations>
                # and keep best (n_configs / eta) configurations
                n_configs = n * self.eta ** (-i)
                n_iters = int(round(r * self.eta ** i))

                results = []
                for config in ListProgress(T, desc='i = %s. Sweeping configs' % i):
                    print('Training %s for %s epochs' % (config.name, n_iters))

                    res_mon = None
                    def get_result_monitor(config):
                        nonlocal res_mon
                        res_mon = HyperbandResultMonitor(config, exp.patience)
                        return res_mon

                    model = Model(exp, config, cpu=args.cpu, debug=args.debug).fit(n_iters, callbacks=[get_result_monitor, ModelSaver])

                    info = dict(reward=res_mon.best_reward, config=config, epoch=res_mon.best_epoch)
                    best_info = max(best_info, info, key=lambda k: k['reward'])
                    if not res_mon.stopped_early:
                        results.append(info)
                    print()
                # select a number of best configurations for the next loop
                results = sorted(results, key=lambda k: k['reward'], reverse=True)
                T = [info['config'] for info in results[: int(n_configs / self.eta)]]

        config, epoch = best_info['config'], best_info['epoch']
        exp.link_best(config.name)
        print('Best config:', config.name)
        print('Iterations:', epoch)
        print(config.load_train_results().loc[epoch].to_string(header=False))

class HyperbandResultMonitor(ResultMonitor): # early stopping based on patience
    def __init__(self, config, patience):
        super(HyperbandResultMonitor, self).__init__(config)
        self.patience = patience
        self.best_reward = -float('inf')
        self.best_epoch = None
        if config.best_reward.exists():
            self.best_reward, self.best_epoch = config.load_best_reward()
        self.stopped_early = False
    
    def on_epoch_end(self, model, train_state):
        super(HyperbandResultMonitor, self).on_epoch_end(model, train_state)
        reward_fn = model.hyperband_reward

        reward = reward_fn(train_state.epoch_result)
        is_best_epoch = reward > self.best_reward
        if is_best_epoch:
            self.best_reward = reward
            self.best_epoch = model.epoch
            self.config.save_best_reward(self.best_reward, self.best_epoch)
            print('New best epoch %s with reward %s' % (self.best_epoch, self.best_reward))
        train_state.save_epoch = is_best_epoch

        if model.epoch > self.patience:
            recent = self.train_results.iloc[-(self.patience + 1):]
            last_rewards = recent.apply(reward_fn, axis=1)
            if last_rewards.idxmax() == recent.index[0]:
                self.stopped_early = model.epoch
                train_state.stop = True
                self.config.set_stopped_early()
                print('Stopped early after %s / %s iterations' % (model.epoch, train_state.stop_epoch))
                return

def cache(f):
    cached_output = None
    def wrapper(*args):
        nonlocal cached_output
        if cached_output is None:
            cached_output = f(*args)
        return cached_output
    return wrapper
