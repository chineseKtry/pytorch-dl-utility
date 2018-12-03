#!/usr/bin/env python2

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from glob import glob
import h5py
import os

from batch_generator import MatrixBatchGenerator

def cont2ohe(old_x,new_x):
    '''converts a sequence in continuous space to one-hot encoded space
       Note: we require old_x (the original one-hot encoded sequence) so that
       we know which indices of the modified sequence to zero out (b/c sequences
       are padded on both ends)
    '''
    ohe_x = np.zeros(old_x.shape)
    max_inds = np.argmax(new_x,axis=0).squeeze()
    ind_sum = np.sum(old_x,axis=0)
    for i in range(old_x.shape[1]):
        ohe_x[max_inds[i],i] = np.round(1*ind_sum[i])
    return np.expand_dims(ohe_x,axis=1)
    # return np.expand_dims(ohe_x*np.sum(old_x,axis=0),axis=1)

class AdversarialGenerator():
    def __init__(self,model,criterion,epsilon=1.0,num_iter=10,single_aa=False,ohe_output=False,use_cuda=False):
        self.network = model.network
        self.epsilon = epsilon
        self.num_iter = num_iter
        self.criterion = criterion
        self.ohe_output = ohe_output
        self.single_aa = single_aa
        self.use_cuda = use_cuda

        self.network.eval()

    def cudafy(self,tensor):
        tensor = tensor.cuda() if self.use_cuda else tensor
        return tensor

    def cont_norm(self,x):
        '''takes as input a continuous representation of a sequence (tensor) and 
        normalizes the values at each position s.t. the sum at each position is 1'''
        norm_x = x/x.abs().sum(dim=0)
        norm_x[norm_x != norm_x] = 0 # set NaN values to zero
        return norm_x

    def single_aa_perturb(self,x,grads):

        new_x = x.copy().squeeze()
        ind_sum = np.sum(x,axis=0).squeeze()
        max_inds = grads.squeeze().reshape(-1).argsort()[-5:]

        for ind in max_inds:
            (i,j) = np.unravel_index(ind,new_x.shape)
            if ind_sum[j] == 1:
                new_x[:,j] = 0
                new_x[i,j] = 1
                break
        return np.expand_dims(new_x,axis=1)

    def generate_classification(self,X,Y=None):
        # note: currently designed only for binary classifcation 
        #       (using nn.CrossEntropyLoss() as criterion)

        seq_tensor = Variable(self.cudafy(torch.FloatTensor(X)),requires_grad=True)
        if Y is None:
            d = self.cudafy(torch.rand(seq_tensor.shape))
            d = d/d.view(d.shape[0],-1).pow(2).sum(1).view(-1,1,1,1)
            seq_tensor.data += d
            target_tensor = Variable(self.cudafy(self.network(seq_tensor)['y_pred_loss']))
        else:
            target_tensor = Variable(self.cudafy(torch.from_numpy(Y[:, 1].astype(np.long))))

        if self.single_aa:
            seq_tensor.grad = None
            out_dict = self.network(seq_tensor)
            loss = self.criterion(out_dict['y_pred_loss'],target_tensor)
            loss.backward()
            grads = seq_tensor.grad.cpu().data.numpy()

            adv_X = np.array([self.single_aa_perturb(X[i],grads[i]) for i in range(X.shape[0])])

            # k = 0
            # for i in range(X.shape[0]):
            #     if np.sum(adv_X[i] != X[0]) == 0:
            #         k += 1
            # print(k,'/',X.shape[0])
            # print(np.sum(adv_X[0] != X[0]))
            # print(np.argmax(adv_X[0].squeeze(),axis=0))
            # print(np.argmax(X[0].squeeze(),axis=0))
            # print('=========')

        else:
            for i in range(self.num_iter):
                # zero out gradients
                seq_tensor.grad = None

                # run forward & backward passes
                out_dict = self.network(seq_tensor)
                loss = self.criterion(out_dict['y_pred_loss'],target_tensor)
                loss.backward()

                # update object
                if Y is None: # maximize loss/divergence for VAT
                    seq_tensor.data += self.epsilon*seq_tensor.grad.data #.sign()
                else: # increase loss/divergence for labeled data (move away from label)
                    seq_tensor.data += self.epsilon*seq_tensor.grad.data #.sign()
                # seq_tensor.data = self.cont_norm(seq_tensor.data)

            adv_X = seq_tensor.cpu().data.numpy()

        if self.ohe_output:
            adv_X = np.array([cont2ohe(X[i].squeeze(),adv_X[i].squeeze()) for i in range(X.shape[0])])

        # print(np.max(adv_X),np.min(adv_X))
        # if np.sum(adv_X != X) > 0:
        # for i in range(adv_X.shape[0]):
        #     if np.sum((np.argmax(adv_X[i].squeeze(),axis=0)-np.argmax(X[i].squeeze(),axis=0))[8:10]) > 0:
            
        #         # if np.sum(adv_X[i] != X[i]) > 0:
        #         print(np.sum(adv_X[i].squeeze() - X[i].squeeze()),np.argmax(adv_X[i].squeeze(),axis=0)[8:10],np.argmax(X[i].squeeze(),axis=0)[8:10])
        #         print(X[i].squeeze()[:,8:10])
        #         print(adv_X[i].squeeze()[:,8:10])
        #         print(np.sum(adv_X[i],0))
        #         # print(adv_X[i])
        #         print('=====')
            # print(np.sum(adv_X != X),np.sum(adv_X[0],axis=0))
        # print(np.sum(adv_X[0],axis=0))

        new_X = np.concatenate([X,adv_X]).astype(np.float32)

        if Y is None:
            return new_X, np.tile(target_tensor.cpu().data.numpy()[:, 0],2)
        else:
            return new_X, np.tile(Y[:, 1].astype(np.long),2)

    def generate_regression(self,X,Y):

        seq_tensor = Variable(self.cudafy(torch.FloatTensor(X)),requires_grad=True)
        target_tensor = Variable(torch.FloatTensor(Y.squeeze()))

        for i in range(self.num_iter):

            # zero out gradients
            seq_tensor.grad = None

            # run forward & backward passes
            out = self.network(seq_tensor)
            loss = self.criterion(out,target_tensor)
            loss.backward()

            # update object
            vals = seq_tensor.grad.data
            seq_tensor.data -= self.epsilon*vals.sign()
            seq_tensor.data = self.cont_norm(seq_tensor.data)

        if self.ohe_output:
            adv_X = np.array([cont2ohe(X[i].squeeze(),seq_tensor.cpu().data.numpy()[i].squeeze()) for i in range(X.shape[0])])
        else:
            adv_X = seq_tensor.cpu().data.numpy()
        
        new_X = np.concatenate([X,adv_X]).astype(np.float32)

        if Y is None:
            return new_X
        else:
            return new_X, np.tile(Y[:, 1].astype(np.long),2)

class AdversarialH5pyBatchGenerator(MatrixBatchGenerator):

    def __init__(self, glob_str, adversarial_generator, batch_size=None, shuffle=False, process_x_y=lambda X, Y: (X, Y)):
        self.adversarial_generator = adversarial_generator

        files = [h5py.File(path) for path in glob(glob_str)]
        X, Y = zip(*[(file['data'][()], file['label'][()]) for file in files])
        X, Y = np.concatenate(X, axis=0), np.concatenate(Y, axis=0)
        X, Y = process_x_y(X, Y)
        super(AdversarialH5pyBatchGenerator, self).__init__(X, Y, batch_size, shuffle)

    # override MatrixBatchGenerator next() to include adversarial examples
    def next(self):
        if self.i >= self.N:
            raise StopIteration
        start = self.i
        self.i += self.batch_size

        if self.Y is None:
            return self.adversarial_generator.generate_classification(self.X[start: self.i])
        else:
            return self.adversarial_generator.generate_classification(self.X[start: self.i], self.Y[start: self.i])
