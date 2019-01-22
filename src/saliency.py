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
	def __init__(self,network,use_cuda=True):
		self.network = network
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
			grads = self.integrated_gradients(seq_ohe,target_label_idx,baseline,steps=steps)

			# sum over each AA/nucleotide position
			# seq_saliency = np.sum(int_grads.squeeze(0)*seq_ohe,axis=0).squeeze()
		elif method == 'grad':
			grads,_ = self.calculate_gradients([seq_ohe],target_label_idx)
			grads = grads.squeeze(0)
		
		seq_saliency = np.sum((grads*seq_ohe).squeeze(),axis=1).squeeze()

		return seq_saliency

	# converts a one-hot encoded sequence to its corresponding string
	def ohe2seq(self,ohe_seq,seqtype='dna'):
		seq = ''
		if seqtype == 'dna':
			# {A:[1,0,0,0],C:[0,1,0,0],G:[0,0,1,0],T:[0,0,0,1]}
			seq_dict = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
			seqinds = np.argmax(ohe_seq,axis=0).squeeze()
			seq = ''.join([seq_dict[ind] for ind in seqinds])

		return seq

	# calculates saliency scores for each sequence in a set of HDF5 files & writes 
	# the results + corresponding sequences to file
	def log_seq_saliency_scores(self,result_dir,data_dir,glob_str,target_label_idx,batch_size=128,steps=10):
		# load datasets
		data_glob_str = os.path.join(data_dir,glob_str)
		files = [h5py.File(path) for path in glob(data_glob_str)]
		X, Y = zip(*[(file['data'][()], file['label'][()]) for file in files])
		X, Y = np.concatenate(X, axis=0), np.concatenate(Y, axis=0)
		X, Y = X[np.where(Y==1)[0]], Y[Y==1]

		# convert ohe-seqs to DNA seqs
		seqs = [self.ohe2seq(X[i],'dna') for i in range(X.shape[0])]

		# calculate saliency scores for all sequences
		sal_scores = np.concatenate([self.seq_saliency_scores(X[i:i+batch_size],target_label_idx,steps=steps) \
			for i in range(0,X.shape[0],batch_size)])

		# write sequences to file
		with open(os.path.join(result_dir,'best_config','saliency.seqs'),'w') as f:
			writer = csv.writer(f,delimiter='\t')
			for i in range(X.shape[0]):
				writer.writerow([seqs[i]])

		# write sequences saliency scores to file
		np.savetxt(os.path.join(result_dir,'best_config','saliency.scores'),sal_scores,delimiter='\t')

	# creates PWMs using a sliding window approach based on saliency scores calculated
	# from log_seq_saliency_scores()
	def create_pwm_arrays_from_grads(self,result_dir,data_dir,window_size=12,batch_size=128):

		from Bio import motifs
		from Bio.Seq import Seq

		# load sequences from file
		with open(os.path.join(result_dir,'best_config','saliency.seqs'),'r') as f:
			reader = csv.reader(f,delimiter='\t')
			seqs = [line[0] for line in reader]

		# identify windows of highest saliency for each sequence & extract subsequences
		salient_seqs = []
		max_scores = []

		with open(os.path.join(result_dir,'best_config','saliency.scores'),'r') as f:
			reader = csv.reader(f,delimiter='\t',quoting=csv.QUOTE_NONNUMERIC)
			for i in range(len(seqs)):
				sal_scores = abs(np.array(reader.next()))
				window_scores = np.convolve(sal_scores,np.ones(window_size,dtype=int),'valid')
				max_ind = np.argmax(window_scores)
				max_score = np.max(window_scores)
				salient_seqs.append(Seq(seqs[i][max_ind:max_ind+window_size]))
				max_scores.append(max_score)

		# filter out low scoring sequences??? skip for now...
		threshold = np.percentile(max_scores,80)
		print('Threshold:' + str(threshold),'Median: ' + str(np.median(max_scores)))
		salient_seqs = [salient_seqs[i] for i in range(len(salient_seqs)) if max_scores[i] >= threshold]

		# create motif from subsequences using BioPython
		with open(os.path.join(result_dir,'best_config','numpy.pwm'),'w') as f:
			writer = csv.writer(f,delimiter=' ')
			motif = motifs.create(salient_seqs)
			for nuc in ['A','C','G','T']:
				writer.writerow(motif.pwm[nuc])

	# creates PWMs using a sliding window approach based on saliency scores calculated
	# from log_seq_saliency_scores() - weights nucleotides for each sequence by the
	# saliency score
	def create_pwm_arrays_from_grads_weighted(self,result_dir,data_dir,window_size=12,batch_size=128):

		from Bio import motifs
		from Bio.Seq import Seq

		# load sequences from file
		with open(os.path.join(result_dir,'best_config','saliency.seqs'),'r') as f:
			reader = csv.reader(f,delimiter='\t')
			seqs = [line[0] for line in reader]

		# identify windows of highest saliency for each sequence & extract subsequences
		salient_seqs = []
		nuc_scores = {nuc: np.zeros(window_size) for nuc in ['A','C','G','T']}
		# max_scores = []

		with open(os.path.join(result_dir,'best_config','saliency.scores'),'r') as f:
			reader = csv.reader(f,delimiter='\t',quoting=csv.QUOTE_NONNUMERIC)
			for i in range(len(seqs)):
				sal_scores = abs(np.array(reader.next()))
				window_scores = np.convolve(sal_scores,np.ones(window_size,dtype=int),'valid')
				max_ind = np.argmax(window_scores)
				max_score = np.max(window_scores)
				for j,nuc in enumerate(seqs[i][max_ind:max_ind+window_size]):
					nuc_scores[nuc][j] += window_scores[max_ind+j]

		# normalize PWM
		for i in range(window_size):
			pos_sum = sum([nuc_scores[nuc][i] for nuc in ['A','C','G','T']])
			for nuc in ['A','C','G','T']:
				nuc_scores[nuc][i] *= 1./pos_sum

		# write PWM to file
		with open(os.path.join(result_dir,'best_config','numpy.pwm'),'w') as f:
			writer = csv.writer(f,delimiter=' ')
			for nuc in ['A','C','G','T']:
				writer.writerow(list(nuc_scores[nuc]))

# creates a saliency map for a DNA sequence given its corresponding saliency scores
def visualize_saliency(seq,saliency_scores):

	import matplotlib.patches as mpatches

	seq_dict = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
	colorIdx_dict = {'A': 'green', 'C': 'blue', 'G':'orange', 'T': 'red'}


	colors = []
	vals = []

	for i in range(len(seq)):
		colors.append(colorIdx_dict[seq[i]])
		vals.append(saliency_scores[i])

	vals_group = []
	for i in range(0,len(vals),5):
		vals_group.append(np.mean(vals[i:i+5]))

	plt.bar(range(1,len(vals)+1),vals,color=colors)
	A_patch = mpatches.Patch(color='green', label='A')
	T_patch = mpatches.Patch(color='red', label='T')
	C_patch = mpatches.Patch(color='blue', label='C')
	G_patch = mpatches.Patch(color='orange', label='G')

	plt.legend(handles=[A_patch,T_patch,C_patch,G_patch])
	plt.xlabel('Nucleotide Position')
	plt.ylabel('Saliency Score')
	plt.show()

