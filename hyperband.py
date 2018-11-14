from __future__ import print_function

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
        s_counter = util.progress_manager.counter(total=self.s_max + 1, desc='Sweeping s', leave=False)
        for s in reversed(xrange(self.s_max + 1)):

            # initial number of configurations
            n = int(math.ceil(self.B / self.max_iter / (s + 1) * self.eta ** s))

            # initial number of iterations per config
            r = self.max_iter * self.eta ** (-s)

            # n random configurations
            T = [Config(self.result_dir, config_dict=self.get_config()) for _ in range(n)]

            i_counter = util.progress_manager.counter(total=s + 1, desc='s = %s. Sweeping i' % s, leave=False)
            for i in xrange(s + 1):
                # Run each of the n configs for <iterations>
                # and keep best (n_configs / eta) configurations
                n_configs = n * self.eta ** (-i)
                n_iterations = int(round(r * self.eta ** i))

                t_counter = util.progress_manager.counter(total=len(T), desc='i = %s. Sweeping configs' % i, leave=False)
                results = []
                for config in T:
                    print('Training', config.name)

                    model = self.get_model(config, self.args)
                    if dry_run:
                        result = {'hyperband_reward': random.random()}
                    else:
                        result = config.get_train_result(n_iterations)

                        if result is not None:
                            print('Loaded previous results')
                            print(result.to_string(header=False, float_format='%.6g'))
                        else:
                            model.fit(self.train_generator, self.val_generator, n_iterations)
                            result = config.get_train_result(n_iterations)

                            assert result is not None, 'Result for every epoch must be saved in the fit loop'

                    assert 'hyperband_reward' in result.index, 'Result must be a dictionary containing the key "hyperband_reward"'
                    print()
                    results.append((config, result))
                    # TODO early stopping
                    t_counter.update()

                # select a number of best configurations for the next loop
                results = sorted(results, key=lambda (config, result): result['hyperband_reward'], reverse=True)
                T = [config for config, result in results[: int(n_configs / self.eta)]]
                self.all_results.extend(results)

                t_counter.close()
                i_counter.update()

            i_counter.close()
            s_counter.update()

        s_counter.close()
        util.progress_manager.stop()
        
        best_config, best_result = max(self.all_results, key=lambda (config, result): result['hyperband_reward'])
        best_config.link_as_best()
        return best_config, best_result
