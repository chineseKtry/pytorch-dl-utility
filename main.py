from __future__ import print_function
import argparse
import numpy as np
import os
from pprint import pprint
import shutil
import sys
import torch

from hyperband import Hyperband
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

parser.add_argument('-m', '--model', dest='model_path',
                    help='Path to python file with PyTorch model')
parser.add_argument('-d', '--data', dest='data_path', help='Data path or directory')
parser.add_argument('-r', '--result', dest='result_dir', help='Result directory')
parser.add_argument('-p', '--predict', dest='predict', help='Prediction input path')

parser.add_argument('-ep', '--epoch', dest='epoch', type=int,
                    help='Number of epochs to train and hyperparameter tune for')
parser.add_argument('-bs', '--batch-size', dest='batch_size', default=100, type=int,
                    help='Batch size in gradient-based training')
args = parser.parse_args()

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
    sys.path.append(args.result_dir)
    import model_def

    best_config_dir = os.path.join(args.result_dir, 'best_config')
    if args.hyper:
        if os.path.islink(best_config_dir):
            print('Best config %s already exists, skipping hyperparameter search' % (best_config_dir))
        else:
            gen = model_def.get_train_generator(args.data_path, args.batch_size)
            if type(gen) == tuple:
                train_generator, val_generator = gen
            else:
                train_generator = gen
                val_generator = model_def.get_val_generator(args.data_path)
            hb = Hyperband(model_def.get_config, model_def.get_model, args.result_dir,
                        train_generator, val_generator, args.epoch, args)
            best_config, best_result = hb.run()
            best_config_name = util.get_config_name(best_config)
            print('Best config:', best_config_name)
            print(best_result.to_string(header=False))
    
    def get_best_model():
        best_config = util.load_json(os.path.join(best_config_dir, 'config.json'))
        model = model_def.get_model(best_config, best_config_dir, args)
        model.load()
        return model

    if args.eval:
        test_result_path = os.path.join(best_config_dir, 'test_result.json')
        if os.path.exists(test_result_path):
            result = util.load_json(test_result_path)
            print('Loaded previous evaluation result:', util.format_json(result))
        else:
            model = get_best_model()
            test_generator = model_def.get_test_generator(args.data_path)
            result = model.evaluate(test_generator)
            print('Evaluation result:', util.format_json(result))
            model.save_test_result(result)

    if args.predict:
        pred_out = os.path.join(best_config_dir, args.predict)
        if os.path.exists(pred_out):
            print('Prediction already exist at %s' % pred_out)
        else:
            model = get_best_model()
            pred_in = os.path.join(args.data_path, args.predict)
            pred_generator = model_def.get_pred_generator(pred_in)
            Y = model.predict(pred_generator)
            np.save(pred_out, Y)
            print('Saved predictions to %s' % pred_out)
