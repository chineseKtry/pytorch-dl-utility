from __future__ import print_function, absolute_import

import argparse
import torch

from hyperband import Hyperband
from experiment import Experiment
from config import Config
from util import *

parser = argparse.ArgumentParser(description='PyTorch + Hyperband for genomics. Run script arguments:', add_help=False)
parser.add_argument('-y', '--hyper', dest='hyper', default=False, action='store_true',
                    help='Perform hyper-parameter tuning')
parser.add_argument('-t', '--train', dest='train', default=False, action='store_true',
                    help='Train a model for a particular hyper-parameter')
parser.add_argument('-e', '--eval', dest='eval', default=False, action='store_true',
                    help='Evaluate model on test set')
parser.add_argument('-p', '--predict', nargs='+', dest='pred', type=Path, help='Prediction input path')
parser.add_argument('--debug', dest='debug', default=False, action='store_true',
                    help='Debug mode, run one iteration')
parser.add_argument('--cpu', dest='cpu', default=False, action='store_true',
                    help='Use CPU instead of GPU')
args, rem_argv = parser.parse_known_args()
if set(rem_argv) & {'--help', '-h'}:
    parser.print_help()

exp_parser = argparse.ArgumentParser(description='Experiment arguments:')
exp_parser.add_argument('-x', '--experiment', dest='exp', type=Path, default='',
                        help='Experiment json file')
exp_parser.add_argument('-c', '--config', dest='config', type=Path, default='',
                        help='Config json file')

exp_parser.add_argument('-r', '--result', dest='res', type=Path, help='Result directory')
exp_parser.add_argument('-f', '--model', dest='model', type=Path,
                    help='Path to python file with PyTorch model')
exp_parser.add_argument('-d', '--data', dest='data', type=Path, help='Data directory')

exp_parser.add_argument('-he', '--hyper-epoch', dest='hyper_epoch', type=int, default=0,
                    help='Number of epochs to tune hyperparameters for')
exp_parser.add_argument('-te', '--train-epoch', dest='train_epoch', type=int, default=0,
                    help='Number of epochs to train for')
exp_parser.add_argument('-bs', '--batch-size', dest='batch_size', type=int, default=100,
                    help='Batch size in gradient-based training')
exp_parser.add_argument('-es', '--early-stopping', dest='early_stopping', default=False, action='store_true',
                    help='Whether to stop early after a number of iterations with no improvement')
exp_parser.add_argument('-pt', '--patience', dest='patience', type=int, default=3,
                    help='Stop training early after this number of iterations with no improvement')
exp_args = exp_parser.parse_args(rem_argv)

if __name__ == '__main__':
    config = None
    if exp_args.exp:
        print('Loading experiment from %s' % exp_args.exp)
        exp = Experiment.from_path(exp_args.exp).set_vars()
        if exp_args.config:
            print('Loading config from %s' % exp_args.config)
            config = Config.from_path(exp_args.config) or exp.config(exp_args.config)
        exp_args = exp_parser.parse_args(exp.get_flags()) # reparse arguments from flags from config
    else:
        exp = Experiment(exp_args.res, exp_args.data, exp_args.model)
    for k, v in vars(exp_args).items():
        if not hasattr(exp, k):
            setattr(exp, k, v)

    if not args.cpu:
        if not torch.cuda.is_available():
            print('GPU not available, switching to CPU')
            args.cpu = True
    if args.debug:
        exp.train_epoch = exp.hyper_epoch = 1

    exp.model.cp(exp.res / 'model_def.py')
    Model = import_module('model', exp.model._).Model

    if args.hyper:
        Hyperband(Model, exp, args).run()
    
    model = None
    if args.train:
        config = config or exp.config(params=Model.get_params(exp))
        print('Training %s for %s epochs' % (config.name, exp.train_epoch))
        model = Model(exp, config, cpu=args.cpu, debug=args.debug)
        model.fit(exp.train_epoch)

    if args.adversarial:

        # start from randomly initialized model
        adv_config = Config(args.result_dir, config_dict=model_def.get_config())
        model = model_def.Model(adv_config,args)

        if args.adversarial == 'normal':
            gen = model_def.get_adversarial_train_generator(args.data_dir, args.batch_size, model, args.epsilon, seq=args.seq) 
        elif args.adversarial == 'fit':
            gen = model_def.get_train_generator(args.data_dir, args.batch_size, seq=args.seq)     

        if type(gen) == tuple:
            train_generator, val_generator = gen
        else:
            train_generator = gen
            val_generator = model_def.get_val_generator(args.data_dir,seq=args.seq)

        if args.adversarial == 'normal' or args.adversarial == 'fit':
            model.fit(train_generator,val_generator, args.epoch) 
            # create symbolic link (automatically treat tested config as best config) 
            if os.path.islink(adv_config.best_dir):
                os.remove(adv_config.best_dir)
            os.symlink(adv_config.name, adv_config.best_dir)

        # virtual adversarial training (still under development...)
        # else:
        #     if not os.path.exists(os.path.dirname(args.adversarial)):
        #         print('Data directory for semi-supervised learning not available.')
        #     else:
        #         print('Performing virtual adversarial training')
        #         virtual_train_generator = model_def.get_virtual_adversarial_train_generator(args.adversarial, args.batch_size, model)
        #         for epoch_no in range(1,args.epoch):
        #             # unsupervised task
        #             model.fit(virtual_train_generator,val_generator, args.epoch)

        #             # supervised task
        #             model.fit(train_generator,val_generator, args.epoch)

    if args.eval:
        config = config or exp.config_best()
        print('Testing %s' % config.name)
        model = model or Model(exp, config, cpu=args.cpu, debug=args.debug)
        model.test()

    if args.pred:
        config = config or exp.config_best()
        print('Predicting with %s on %s' % (config.name, args.pred))
        model = model or Model(exp, config, cpu=args.cpu, debug=args.debug)
        for p in args.pred:
            model.predict(p)
        