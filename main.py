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
parser.add_argument('-pi', '--predict', dest='predict_path', help='Prediction data path')

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

    shutil.copy(args.model_path, os.path.join(args.result_dir, 'model_def.py'))
    sys.path.append(args.result_dir)
    import model_def

    if args.hyper:
        train_generator = model_def.get_train_generator(args.data_path, args.batch_size)
        val_generator = model_def.get_val_generator(args.data_path)
        hb = Hyperband(model_def.get_config, model_def.get_model, args.result_dir,
                       train_generator, val_generator, args.epoch, args)
        best_config, best_result = hb.run()
        best_config_name = util.get_config_name(best_config)
        print('Best config: %s\n Result: %s' % (best_config_name, util.format_json(best_result)))

    if args.eval or args.predict:
        best_config_dir = os.path.join(args.result_dir, 'best_config')
        best_config = util.load_json(os.path.join(best_config_dir, 'config.json'))

        model = model_def.get_model(best_config, best_config_dir, args)
        model.load()

    if args.eval:
        test_generator = model_def.get_test_generator(args.data_path)

        result = model.evaluate(test_generator)
        print('Evaluation result', util.format_json(result))

        util.save_json(result, os.path.join(best_config_dir, 'evaluation.json'))

    if args.predict_path: # TODO not sure how this is supposed to look like yet
        pred_generator = model_def.get_predict_generator(args.predict_path)

        Y = model.predict(pred_generator)
        model_def.save_prediction(Y, args.predict_path)
