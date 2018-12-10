#!/usr/bin/env python2

import torch
from torch import nn
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
import os
from glob import glob
import csv

class SaliencyMap():
	def __init__(self,model,use_cuda=True):
		self.network = model
		self.use_cuda = use_cuda

		if use_cuda:
			self.network.to('cuda')

	def cudafy(self,tensor):
		tensor = tensor.cuda() if self.use_cuda else tensor
		return tensor

	def calculate_gradients(self,inputs, target_label_idx):

		gradients = []
		for inp in inputs:
			inp_tensor = self.cudafy(torch.FloatTensor(inp))
			inp_tensor.requires_grad=True
			out_dict = self.network(inp_tensor)
			out = out_dict['y_pred_loss'] # logits

			if target_label_idx is None:
				target_label_idx = torch.argmax(out, 1).item()

			index = np.ones((out.size()[0], 1)) * target_label_idx
			index = self.cudafy(torch.tensor(index, dtype=torch.int64))

			out_target = out.gather(1,index).sum()
			self.network.zero_grad()
			out_target.backward()
			gradients.append(inp_tensor.grad.data.cpu().numpy())

		return np.array(gradients), target_label_idx

	# integrated gradients
	def integrated_gradients(self,inputs, target_label_idx, baseline, steps=50):

		if baseline is None:
			baseline = 0 * inputs 
		# scale inputs and compute gradients
		scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(0, steps + 1)]
		grads, _ = self.calculate_gradients(scaled_inputs, target_label_idx)	
		avg_grads = np.average(np.swapaxes(grads,0,1), axis=1)
		integrated_grad = (inputs - baseline) * avg_grads
		return integrated_grad

	# calculates gradients or integrated gradients for a given set of sequences + labels
	def seq_saliency_scores(self,seq_ohe,target_label_idx,method='int_grad',steps=10):

		baseline = np.ones(seq_ohe.shape)/float(seq_ohe.shape[-3])
		if method == 'int_grad':
			int_grads = self.integrated_gradients(seq_ohe,target_label_idx,baseline,steps=steps)
			# sum over each AA/nucleotide position
			# seq_saliency = np.sum(int_grads.squeeze(0)*seq_ohe,axis=0).squeeze()
			seq_saliency = np.sum((int_grads*seq_ohe).squeeze(),axis=1).squeeze()
		elif method == 'grad':
			seq_saliency = self.calculate_gradients([[seq_ohe]],target_label_idx)

		return seq_saliency

	def conv_feature_weights(self):
		return self.network.state_dict()['features.0.weight']

	def ohe2seq(self,ohe_seq,seqtype='dna'):
		seq = ''
		if seqtype == 'dna':
			# {A:[1,0,0,0],C:[0,1,0,0],G:[0,0,1,0],T:[0,0,0,1]}
			seq_dict = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
			seqinds = np.argmax(ohe_seq,axis=0).squeeze()
			seq = ''.join([seq_dict[ind] for ind in seqinds])

		return seq
			
	def create_pwm_arrays_from_grads(self,result_dir,data_dir,glob_str,target_label_idx,window_size=8,batch_size=128):

		from Bio import motifs
		from Bio.Seq import Seq

		# load datasets
		data_glob_str = os.path.join(data_dir,glob_str)
		files = [h5py.File(path) for path in glob(data_glob_str)]
		X, Y = zip(*[(file['data'][()], file['label'][()]) for file in files])
		X, Y = np.concatenate(X, axis=0), np.concatenate(Y, axis=0)
		X, Y = X[np.where(Y==0)[0]], Y[Y==0]

		# convert ohe-seqs to DNA seqs
		seqs = [self.ohe2seq(X[i],'dna') for i in range(X.shape[0])]

		# calculate saliency scores for all sequences
		sal_scores = np.concatenate([self.seq_saliency_scores(X[i:i+batch_size],target_label_idx) \
			for i in range(0,X.shape[0],batch_size)])

		# identify windows of highest saliency for each sequence & extract subsequences
		salient_seqs = []
		max_scores = []
		for i in range(sal_scores.shape[0]):
			max_score = -100
			max_ind = 0
			for j in range(0,sal_scores.shape[1]-window_size):
				if sum(sal_scores[i,j:j+window_size]) > max_score:
					max_ind = j
					max_score = sum(sal_scores[i,j:j+window_size])
			salient_seqs.append(Seq(seqs[i][max_ind:max_ind+window_size]))
			max_scores.append(max_score)

		# # filter out low scoring sequences??? skip for now...
		# threshold = np.percentile(max_scores,80)
		# print('Threshold:' + str(threshold),np.median(max_scores))
		# salient_seqs = [salient_seqs[i] for i in range(len(salient_seqs)) if max_scores[i] >= threshold]

		# create motif from subsequences using BioPython
		f = open(os.path.join(result_dir,'best_config','numpy_flip.pwm'),'w')
		writer = csv.writer(f,delimiter=' ')
		motif = motifs.create(salient_seqs)
		for nuc in ['A','C','G','T']:
			writer.writerow(motif.pwm[nuc])
		f.close()

if __name__ == '__main__':
	print('starting...')
	result_dir = '/cluster/alexwu/adversarial/test-adv/saliency/tfbs-results/adversarial/wgEncodeAwgTfbsSydhK562Bhlhe40nb100IggrabUniPk/'
	sys.path.append(result_dir)

	import model_def

	config = model_def.get_config()
	model = model_def.Network_32x2_16_dna(config)

	save_path = os.path.join(result_dir,'best_config','models','model-3.pth')
	state = torch.load(save_path) #,map_location='cpu')
	model.load_state_dict(state['network'])

	sm = SaliencyMap(model)

	data_dir = "/cluster/zeng/research/recomb/generic/saber/wgEncodeAwgTfbsSydhK562Bhlhe40nb100IggrabUniPk/CV0/data/"

	glob_str = 'test.h5.batch*'
	target_label_idx = 0
	sm.create_pwm_arrays_from_grads(result_dir,data_dir,glob_str,target_label_idx,window_size=8,batch_size=256)


# def calculate_outputs_and_gradients(inputs, model, target_label_idx, cuda=False):
#     # do the pre-processing
#     predict_idx = None
#     gradients = []
#     for input in inputs:
#         input = pre_processing(input, cuda)
#         output = model(input)
#         output = F.softmax(output, dim=1)
#         if target_label_idx is None:
#             target_label_idx = torch.argmax(output, 1).item()
#         index = np.ones((output.size()[0], 1)) * target_label_idx
#         index = torch.tensor(index, dtype=torch.int64)
#         if cuda:
#             index = index.cuda()
#         output = output.gather(1, index)
#         # clear grad
#         model.zero_grad()
#         output.backward()
#         gradient = input.grad.detach().cpu().numpy()[0]
#         gradients.append(gradient)
#     gradients = np.array(gradients)
#     return gradients, target_label_idx