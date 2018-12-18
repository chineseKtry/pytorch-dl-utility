from __future__ import print_function, absolute_import

import argparse
from glob import glob
import numpy as np
import os
from pprint import pprint
import shutil
import sys
import torch

from hyperband import Hyperband
from config import Config
import util

parser = argparse.ArgumentParser(description='PyTorch + Hyperband for genomics')
parser.add_argument('-y', '--hyper', dest='hyper', default=False, action='store_true',
                    help='Perform hyper-parameter tuning')
parser.add_argument('-e', '--eval', dest='eval', default=False, action='store_true',
                    help='Evaluate model on test set')
parser.add_argument('--debug', dest='debug', default=False, action='store_true',
                    help='Debug mode, run one iteration')
parser.add_argument('-c', '--cpu', dest='cpu', default=False, action='store_true',
                    help='Use CPU instead of GPU')

parser.add_argument('-f', '--model', dest='model_path',
                    help='Path to python file with PyTorch model')
parser.add_argument('-d', '--data', dest='data_dir', help='Data directory')
parser.add_argument('-r', '--result', dest='result_dir', help='Result directory')
parser.add_argument('-p', '--predict', dest='pred_subpath', help='Prediction input path')

parser.add_argument('-ep', '--epoch', dest='epoch', type=int,
                    help='Number of epochs to train and hyperparameter tune for')
parser.add_argument('-bs', '--batch-size', dest='batch_size', default=100, type=int,
                    help='Batch size in gradient-based training')
parser.add_argument('-es', '--early-stopping', dest='early_stopping', default=False, action='store_true',
                    help='Whether to stop early after a number of iterations with no improvement')
parser.add_argument('-adv','--adversarial',dest='adversarial',help='Perform adversarial training')
parser.add_argument('-epsilon','--epsilon',dest='epsilon',default=1.0,type=float,
                    help='Epsilon value for creating adversarial perturbations')
parser.add_argument('-seq','--seq',dest='seq',default='aa',type=str,
                    help='Sequence type (aa, dna, etc.)')

args = parser.parse_args()

def import_model(path):
    import imp
    model_def = imp.load_source('model', path)
    return model_def

if __name__ == '__main__':
    if not args.cpu:
        if not torch.cuda.is_available():
            print('GPU not available, switching to CPU')
            args.cpu = True

    if args.debug:
        args.epoch = 1
        args.batch_size = 2

    util.makedirs(args.result_dir)
    shutil.copy(args.model_path, os.path.join(args.result_dir, 'model_def.py'))
    model_def = import_model(args.model_path)

    best_config = Config(args.result_dir, from_best=True)
    if args.hyper:
        if best_config.exist_best():
            print('Best config %s => %s already exists, skipping hyperparameter search' % (best_config.best_dir, best_config.name))
        else:
            gen = model_def.get_train_generator(args.data_dir, args.batch_size)
            if type(gen) == tuple:
                train_generator, val_generator = gen
            else:
                train_generator = gen
                val_generator = model_def.get_val_generator(args.data_dir)
            hb = Hyperband(model_def.get_config, model_def.Model, args.result_dir,
                        train_generator, val_generator, args.epoch, args)
            best_result = hb.run()
            best_config = best_result['config']
            print('Best config:', best_config.name)
            print('Iterations:', best_result['epochs'])
            print(best_result['result'].to_string(header=False))

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
        else:
            if not os.path.exists(os.path.dirname(args.adversarial)):
                print('Data directory for semi-supervised learning not available.')
            else:
                print('Performing virtual adversarial training')
                virtual_train_generator = model_def.get_virtual_adversarial_train_generator(args.adversarial, args.batch_size, model)
                for epoch_no in range(1,args.epoch):
                    # unsupervised task
                    model.fit(virtual_train_generator,val_generator, args.epoch)

                    # supervised task
                    model.fit(train_generator,val_generator, args.epoch)

    if args.eval:
        best_config.test_result_path = os.path.join(best_config.save_dir, 'test_result.json')
        result = best_config.load_test_result()
        if result is not None:
            print('Loaded previous evaluation result:', util.format_json(result))
        else:
            model = model_def.Model(best_config, args).load()
            test_generator = model_def.get_test_generator(args.data_dir,seq=args.seq)
            # test_generator = model_def.get_adversarial_test_generator(args.data_dir, args.batch_size, model, args.epsilon, seq=args.seq)
            result = model.evaluate(test_generator)
            print('Evaluation result:', util.format_json(result))
            best_config.save_test_result(result)

    if args.pred_subpath:
        pred_in_glob = os.path.join(args.data_dir, args.pred_subpath)
        model = model_def.Model(best_config, args).load()
        for pred_in in glob(pred_in_glob):
            subpath = pred_in.replace(args.data_dir, '').lstrip('/')
            pred_out = os.path.join(best_config.save_dir, subpath) + '.npy'
            if os.path.exists(pred_out):
                print('Prediction already exist at %s' % pred_out)
            else:
                pred_generator = model_def.get_pred_generator(pred_in)
                Y = model.predict(pred_generator)
                if hasattr(model_def, 'save_prediction'):
                    model_def.save_prediction(args.pred_subpath, Y)
                else:
                    np.save(pred_out, Y)
                    print('Saved predictions to %s' % pred_out)
