from __future__ import print_function, absolute_import

import argparse
import sys
import os

from src.config import Config
from src.saliency import SaliencyMap

parser = argparse.ArgumentParser(description='Saliency for DL + Genomics')
parser.add_argument('result', type=Path, help='Result directory')
parser.add_argument('-pe', '--pred-epoch', dest='pred_epoch', type=int, help='Epoch to test with. If not specified, will use the best performing epoch')

parser.add_argument('-bs', '--batch-size', dest='batch_size', default=100, type=int,
                    help='Batch size in gradient-based saliency maps')
parser.add_argument('-ws', '--window-size', dest='window_size', default=12, type=int, help='Window size for motif detection')
parser.add_argument('-st', '--steps', dest='steps', default=10, type=int, help='Number of steps for integrated gradients')
parser.add_argument('-w','--weighted',dest='weighted',default=False,help='Gradients-based weighting for PWM extraction')

parser.add_argument('--cpu', dest='cpu', default=False, action='store_true',
                    help='Use CPU instead of GPU')
args = parser.parse_args()


if __name__ == '__main__':
	config = Config(args.result)

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
	
	# log saliency scores
	sm = SaliencyMap(model.network)
	glob_str = 'test.h5.batch*'
	target_label_idx = 0
	sm.log_seq_saliency_scores(config.res,config.data,glob_str,target_label_idx,batch_size=args.batch_size)

	# create PWMs
	if args.weighted:
		sm.create_pwm_arrays_from_grads_weighted(config.res,config.data,window_size=args.window_size,batch_size=args.batch_size)
	else:
		sm.create_pwm_arrays_from_grads(config.res,config.data,window_size=args.window_size,batch_size=args.batch_size)
