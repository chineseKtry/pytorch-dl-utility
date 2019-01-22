from __future__ import print_function, absolute_import
from six.moves import range

import argparse
from collections import namedtuple
import math

from train import get_train_args
from src.util import *
from src.util.progress_bar import ListProgress, RangeProgress
from src.config import Config
from src.callbacks.model_saver import ModelSaver
from src.callbacks.result_monitor import ResultMonitor


class Hyperband(object):
    def __init__(self, args, config_dict):
        self.args = args
        self.hb_path = args.hyperband_path

        if self.config_path.exists():
            config = self.load_config()
            args.model_path = Path(config['model'])
            args.data_path = Path(config['data'])
            args.train_epoch = config['train_epoch']
            config_dict.update(config['vars'])
        
        assert args.model_path, 'Must provide "model" argument'
        assert args.data_path, 'Must provide "data" argument'
        assert args.train_epoch, 'Must provide "train-epoch" argument'

        self.Model = import_module('model', args.model_path._).Model
        self.config_dict = config_dict
        
        self.max_iter = args.train_epoch  # maximum iterations per configuration
        self.eta = math.e  # defines configuration downsampling rate

        logeta = lambda x: math.log(x) / math.log(self.eta)
        self.s_max = int(logeta(self.max_iter))
        self.B = (self.s_max + 1) * self.max_iter

    def run(self):
        Model = self.Model
        args = self.args

        best_path = self.best_path
        if best_path.exists():
            print('Best config %s => %s already exists, terminating Hyperband' % (best_path._name, best_path._real._name))
            return
        
        state = self.load_state()
        if state:
            print('Loaded Hyperband initial parameters to search from %s' % self.state_path)
        else:
            config = Config(None, **self.config_dict)
            state = {}
            for s in range(self.s_max, -1, -1):
                # initial number of configurations
                n = int(math.ceil(self.B / self.max_iter / (s + 1) * self.eta ** s))

                # n random configurations
                state[s] = [Model.get_params(config) for _ in range(n)]
            self.save_state(state)
            print('Generated Hyperband initial parameters to search to %s' % self.state_path)

        get_name = lambda params: ','.join(sorted('%s=%s' % kv for kv in params.items()))
        best_info = dict(reward=-float('inf'))
        for s in RangeProgress(self.s_max, -1, step=-1, desc='Sweeping s'):
            T = []
            for params in state[s]:
                name = self.hb_path / get_name(params)
                params.update(self.config_dict)
                t = Config(name, **params)
                t.save(force=True, model=args.model_path, data=args.data_path)
                T.append(t)
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
                        res_mon = HyperbandResultMonitor(config, config.get('early_stop', None))
                        return res_mon

                    model = Model(config, cpu=args.cpu, debug=args.debug).fit(n_iters, callbacks=[get_result_monitor, ModelSaver])

                    info = dict(reward=res_mon.best_reward, config=config, epoch=res_mon.best_epoch)
                    best_info = max(best_info, info, key=lambda k: k['reward'])
                    if not res_mon.stopped_early:
                        results.append(info)
                    print()
                # select a number of best configurations for the next loop
                results = sorted(results, key=lambda k: k['reward'], reverse=True)
                T = [info['config'] for info in results[: int(n_configs / self.eta)]]

        config, epoch = best_info['config'], best_info['epoch']
        self.link_best(config.name)
        print('Best config:', config.name)
        print('Iterations:', epoch)
        print(config.load_train_results().loc[epoch].to_string(header=False))


    @property
    def config_path(self):
        return self.hb_path / 'hyperband_config.json'
    
    def load_config(self):
        return load_json(self.config_path)


    @property
    def best_path(self):
        return self.hb_path / 'best_config'
    
    def link_best(self, name):
        self.best_path.link(name)
    

    @property
    def state_path(self):
        return self.hb_path / 'hyperband_state.json'

    def load_state(self):
        if self.state_path.exists():
            return { int(k): v for k, v in load_json(self.state_path).items() }
        return None

    def save_state(self, state):
        save_json(self.state_path, numpy_to_builtin(state))


class HyperbandResultMonitor(ResultMonitor): # early stopping based on patience
    def __init__(self, config, early_stop):
        super(HyperbandResultMonitor, self).__init__(config)
        self.early_stopping = early_stop or False
        self.patience = early_stop
        self.best_reward = -float('inf')
        self.best_epoch = None
        if config.best_reward.exists():
            self.best_reward, self.best_epoch = config.load_best_reward()
        self.stopped_early = False
    
    def on_epoch_end(self, model, train_state):
        super(HyperbandResultMonitor, self).on_epoch_end(model, train_state)
        reward_fn = model.hyperband_reward

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

        if self.early_stopping and model.epoch > self.patience:
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


parser = argparse.ArgumentParser(description='Search model parameter space with Hyperband')
parser.add_argument('hyperband_path', type=Path, help='Hyperband directory')
parser.add_argument('--clean', type=int, help='1 = remove the configs that are not best, 2 = remove all files except hyperband_config.json')

if __name__ == '__main__':
    args, config_dict = get_train_args(parser)

    if args.clean:
        if args.clean == 2:
            subdirs, files = args.hyperband_path.ls()
            for subdir in subdirs:
                subdirs.rm()
            for f in files:
                if f._name != 'hyperband_config.json':
                    f.rm()
        exit()

    hb = Hyperband(args, config_dict)

    hb.run()
    