from __future__ import print_function, absolute_import

import argparse
import torch

from src.config import Config
from src.util import *

parser = argparse.ArgumentParser(description='Model training')
parser.add_argument('result', type=Path, help='Result directory')

def get_train_args(parser, config_arg_names=['train_batch', 'early_stop']):
    parser.add_argument('-f', '--model', dest='model', type=Path,
                        help='Path to python model file')
    parser.add_argument('-d', '--data', dest='data', type=Path, help='Data directory')

    parser.add_argument('-te', '--train-epoch', dest='train_epoch', type=int, help='Number of epochs to train for')
    parser.add_argument('-tb', '--train-batch', dest='train_batch', type=int,
                        help='Batch size in gradient-based training')
    parser.add_argument('-es', '--early-stop', dest='early_stop', type=int, default=0,
                        help='Whether to stop early after a number of iterations with no improvement to the reward criteria')

    parser.add_argument('--debug', dest='debug', default=False, action='store_true',
                        help='Debug mode, run one iteration')
    parser.add_argument('--cpu', dest='cpu', default=False, action='store_true',
                        help='Use CPU instead of GPU')
    
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print('GPU not available, switching to CPU')
        args.cpu = True
    
    if args.debug:
        args.train_epoch = 1

    config_dict = {
        k: v for k, v in vars(args).items() if v is not None and k in config_arg_names
    }

    return args, config_dict

def train(config, train_epoch, cpu=False, debug=False):
    assert config.model.exists(), 'No model is specified or linked'
    assert config.data.exists(), 'No data is specified or linked'

    Model = import_module('model', config.model._real._).Model
    print('Training %s for %s epochs' % (config.name, train_epoch))
    model = Model(config, cpu=cpu, debug=debug)
    assert train_epoch is not None, 'Train epoch cannot be none'
    model.fit(train_epoch)


if __name__ == '__main__':
    args, config_dict = get_train_args(parser)

    config = Config(args.result, **config_dict)
    config.save(force=True, model=args.model, data=args.data)
    
    train(config, args.train_epoch, cpu=args.cpu, debug=args.debug)
