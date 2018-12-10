import argparse
import sys
import os
import torch
from saliency import SaliencyMap

parser = argparse.ArgumentParser(description='Saliency for DL + Genomics')
parser.add_argument('-r', '--result', dest='result_dir', help='Result directory')
parser.add_argument('-d', '--data', dest='data_dir', help='Data directory')
parser.add_argument('-bs', '--batch-size', dest='batch_size', default=100, type=int,
                    help='Batch size in gradient-based saliency maps')
parser.add_argument('-ws', '--window-size', dest='window_size', default=8, type=int, help='Window size for motif detection')

args = parser.parse_args()


if __name__ == '__main__':

	sys.path.append(args.result_dir)

	import model_def

	config = model_def.get_config()
	model = model_def.Network_32x2_16_dna(config)

	save_path = os.path.join(args.result_dir,'best_config','models','model-3.pth')
	state = torch.load(save_path) #,map_location='cpu')
	model.load_state_dict(state['network'])

	sm = SaliencyMap(model)

	glob_str = 'test.h5.batch*'
	target_label_idx = 0
	sm.create_pwm_arrays_from_grads(args.result_dir,args.data_dir,glob_str,target_label_idx,\
		window_size=args.window_size,batch_size=args.batch_size)