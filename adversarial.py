#!/usr/bin/env python2

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

def cont2ohe(old_x,new_x):
    '''converts a sequence in continuous space to one-hot encoded space
       Note: we require old_x (the original one-hot encoded sequence) so that
       we know which indices of the modified sequence to zero out (b/c sequences
       are padded on both ends)
    '''
    ohe_x = np.zeros(old_x.shape)
    max_inds = np.argmax(x,axis=0).squeeze()
    for i in range(old_x.shape[1]):
        ohe_x[max_inds[i],i] = 1

    return np.expand_dims(ohe_x*np.sum(old_x,axis=0),axis=1)

class AdversarialBatchGenerator():
    def __init__(self,model,criterion,epsilon=0.01,num_iter=10,ohe_output=False,use_cuda=False):
        self.model = model
        self.epsilon = epsilon
        self.num_iter = num_iter
        self.criterion = criterion
        self.ohe_output = ohe_output
        self.use_cuda = use_cuda

        self.model.eval()

    def cudafy(self,tensor):
        tensor = tensor.cuda() if self.use_cuda else tensor
        return tensor

    def generate_classification(self,X,Y):
        # note: currently designed only for binary classifcation 
        #       (using nn.CrossEntropyLoss() as criterion)

        seq_tensor = Variable(self.cudafy(torch.FloatTensor(X)),requires_grad=True)
        target_tensor = Variable(torch.from_numpy(Y[:, 1].astype(np.long)))

        for i in range(self.num_iter):

            # zero out gradients
            seq_tensor.grad = None

            # run forward & backward passes
            out_dict = self.model(seq_tensor)
            loss = self.criterion(out_dict['y_pred_loss'],target_tensor)
            loss.backward()

            # update object
            vals = seq_tensor.grad.data
            seq_tensor.data -= self.epsilon*vals.sign()

        if self.ohe_output:
            adv_X = np.array([cont2ohe(X[i].squeeze(),seq_tensor.data.numpy()[i].squeeze()) for i in range(X.shape[0])])
            return np.concatenate([X,adv_X]), np.tile(Y[:, 1].astype(np.long),2)
        else:
            adv_X = seq_tensor.data.numpy()
            return np.concatenate([X,adv_X]), np.tile(Y[:, 1].astype(np.long),2)

    def generate_regression(self,X,Y):

        seq_tensor = Variable(self.cudafy(torch.FloatTensor(X)),requires_grad=True)
        target_tensor = Variable(torch.FloatTensor(Y.squeeze()))

        for i in range(self.num_iter):

            # zero out gradients
            seq_tensor.grad = None

            # run forward & backward passes
            out = self.model(seq_tensor)
            loss = self.criterion(out,target_tensor)
            loss.backward()

            # update object
            vals = seq_tensor.grad.data
            seq_tensor.data -= self.epsilon*vals.sign()

        if self.ohe_output:
            adv_X = np.array([cont2ohe(X[i].squeeze(),seq_tensor.data.numpy()[i].squeeze()) for i in range(X.shape[0])])
            return np.concatenate([X,adv_X]), np.tile(torch.FloatTensor(Y),2)
        else:
            adv_X = seq_tensor.data.numpy()
            return np.concatenate([X,adv_X]), np.tile(torch.FloatTensor(Y),2)