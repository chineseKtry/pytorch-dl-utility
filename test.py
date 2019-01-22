from __future__ import print_function, absolute_import

import argparse
import torch

from src.config import Config
from src.util import *

parser = argparse.ArgumentParser(description='Model evaluation')
parser.add_argument('result', type=Path, help='Result directory')
parser.add_argument('-v', '--eval', dest='eval', default=False, action='store_true', help='Evaluate model on test set')
parser.add_argument('-pi', '--pred-in', nargs='+', dest='pred_in', type=Path, help='Prediction input paths')
parser.add_argument('-po', '--pred-out', nargs='+', dest='pred_out', type=Path, help='Prediction output paths. If not specified, will be basename(pred_in).npy')

parser.add_argument('-pe', '--pred-epoch', dest='pred_epoch', type=int, help='Epoch to test with. If not specified, will use the best performing epoch')
parser.add_argument('-pb', '--pred-batch', dest='pred_batch', type=int,
                    help='Batch size to evaluate with')

parser.add_argument('--cpu', dest='cpu', default=False, action='store_true',
                    help='Use CPU instead of GPU')

args = parser.parse_args()

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print('GPU not available, switching to CPU')
        args.cpu = True
    
    args = parser.parse_args()
    config = Config(args.result, **{
        k: v for k, v in vars(args).items() if v is not None and k in ['pred_epoch', 'pred_batch']
    })

    assert config.model.exists(), 'No model is linked'
    assert config.data.exists(), 'No data is linked'

    if not args.pred_epoch:
        state = config.load_best_model_state()
    else:
        state = config.load_model_state(epoch=args.pred_epoch)
    assert state is not None, 'No saved trained model exist'

    Model = import_module('model', config.model._real._).Model
    model = Model(config, cpu=args.cpu)
    model.set_state(state)

    if args.eval:
        result = config.load_test_result(epoch=args.pred_epoch)
        if result:
            print('Loaded previous test result:', format_json(result))
        else:
            result = model.evaluate(model.get_test_data())
            config.save_test_result(result, epoch=args.pred_epoch)
            print('Test result:', format_json(result))

    if args.pred_in:
        pred_in, pred_out = args.pred_in, args.pred_out
        if not pred_out:
            pred_out = []
            for pi in pred_in:
                po_name = pi._name + (('-%s' % args.pred_epoch) if args.pred_epoch else '') + '.npy'
                pred_out.append(config.res / po_name)
            assert len(pred_out) == len(set(pred_out)), 'Multiple files have the same name, specify output files manually with the "pred-out" argument'

        assert len(pred_out) == len(pred_in), 'Arguments for "pred-out" has different length than argment for "pred-in"'

        for pi, po in zip(pred_in, pred_out):
            if po.exists():
                print('Prediction %s for %s already exists, skipping' % (po, pi))
                continue
            pred = model.predict(model.get_pred_data(pi))
            model.save_pred(po, pred)
            print('Saved prediction for %s to %s' % (pi, po))
