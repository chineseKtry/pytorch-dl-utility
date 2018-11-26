from __future__ import print_function, absolute_import
from six.moves import range

import math
import os
import random
import time

from config import Config
import util


class Hyperband:

    def __init__(self, get_config, model, result_dir, train_generator, val_generator, max_iter, args, eta=3):
        self.get_config = get_config
        self.get_model = model
        self.result_dir = result_dir
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.max_iter = max_iter  # maximum iterations per configuration
        self.args = args
        self.eta = eta  # defines configuration downsampling rate (default = 3)

        self.logeta = lambda x: math.log(x) / math.log(self.eta)
        self.s_max = int(self.logeta(self.max_iter))
        self.B = (self.s_max + 1) * self.max_iter
        self.all_results = []

    def run(self, dry_run=False):
        args = self.args
        s_counter = util.progress_manager.counter(total=self.s_max + 1, desc='Sweeping s', leave=False)
        for s in reversed(range(self.s_max + 1)):

            # initial number of configurations
            n = int(math.ceil(self.B / self.max_iter / (s + 1) * self.eta ** s))

            # initial number of iterations per config
            r = self.max_iter * self.eta ** (-s)

            # n random configurations
            T = [Config(self.result_dir, config_dict=self.get_config()) for _ in range(n)]

            i_counter = util.progress_manager.counter(total=s + 1, desc='s = %s. Sweeping i' % s, leave=False)
            for i in range(s + 1):
                # Run each of the n configs for <iterations>
                # and keep best (n_configs / eta) configurations
                n_configs = n * self.eta ** (-i)
                n_iterations = int(round(r * self.eta ** i))

                t_counter = util.progress_manager.counter(total=len(T), desc='i = %s. Sweeping configs' % i, leave=False)
                results = []
                for config in T:
                    print('Training', config.name)

                    model = self.get_model(config, args)
                    if dry_run:
                        result = {'hyperband_reward': random.random()}
                        stopped_early = False
                    else:
                        result = config.get_train_result(n_iterations)

                        if result is not None:
                            print('Loaded previous results')
                            print(result.to_string(header=False, float_format='%.6g'))
                        else:
                            epoch, result, stopped_early = model.fit(self.train_generator, self.val_generator, n_iterations, early_stopping=args.early_stopping)
                            if stopped_early and epoch < n_iterations:
                                print('Stopped early at iteration %s' % epoch)

                    assert 'hyperband_reward' in result.index, 'Result must be a dictionary containing the key "hyperband_reward"'
                    print()
                    results.append({
                        'config': config,
                        'epochs': epoch,
                        'result': result,
                        'stopped_early': stopped_early
                    })
                    t_counter.update()

                self.all_results.extend(results)
                # select a number of best configurations for the next loop
                results = [result for result in results if not result['stopped_early']]
                results = sorted(results, key=lambda result: result['result']['hyperband_reward'], reverse=True)
                T = [result['config'] for result in results[: int(n_configs / self.eta)]]

                t_counter.close()
                i_counter.update()

            i_counter.close()
            s_counter.update()

        s_counter.close()
        util.progress_manager.stop()
        
        best_result = max(self.all_results, key=lambda result: result['result']['hyperband_reward'])
        best_result['config'].link_as_best()
        return best_result
