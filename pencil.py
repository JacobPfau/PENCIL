import torch
from torch import nn
from torch import optim
import numpy as np
import copy as cp

class PENCIL():
    def __init__(self, n_samples, n_classes, n_epochs, lrs, alpha, beta, gamma, K=10, save_losses=False, use_KL=True):
        '''
        n_samples: int, length of training dataset
        n_epochs: list of positive ints, number of epochs of phases in form [n_epochs_i for i in range(3)]
        lrs: list of floats, learnings rates for phases in form [lr_i for i in range(3)]
        alpha: coefficient for lo loss
        beta: coefficient for le loss
        gamma: coefficient for label estimate update
        K: int, learning rate multiplier for label estimate updates
        save_losses: bool, whether to save losses into list of lists of form [[lc,lo,le] for e in *phase 2 epochs*]
        use_KL: bool, whether to use KL loss or crossentropy for phase 3
        '''
        
        self.save_losses = save_losses
        self.use_KL = use_KL
        self.n_epochs = n_epochs
        self.lrs = lrs
        self.n_classes = n_classes
        assert self.n_epochs[0]>0, 'Phase 0 must be non-empty to initialize y_tilde'
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.K = K
        
        self.CELoss = nn.CrossEntropyLoss()
        self.KLLoss = nn.KLDivLoss(reduction='mean') #PENCIL official implementation uses mean, not batchmean
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
        self.y_tilde = np.zeros([n_samples,n_classes])
        self.y_prev = None
        self.losses = []
        
    def set_lr(self, optimizer, epoch):
        '''
        Call before inner training loop to update lr based on PENCIL phase
        '''
        lr = -1
        if epoch == 0: lr = self.lrs[0]
        if epoch == self.n_epochs[0]: lr = self.lrs[1]
        elif epoch == self.n_epochs[0]+self.n_epochs[1]: lr = self.lrs[2]
        
        if lr!=-1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
    def get_loss(self, epoch, outputs, labels, indices):
        '''
        labels: cuda tensor of noisy labels
        indices: cpu tensor of indices for current batch
        '''
        # Calculate loss based on current phase
        if epoch < self.n_epochs[0]: #Phase 1
            lc = self.CELoss(outputs, labels)
            # init y_tilde as we train
            cpu_labels = labels.cpu()
            labels_temp = torch.zeros(cpu_labels.size(0), self.n_classes).scatter_(1, cpu_labels.view(-1, 1), self.K) #apply temperature 10 before softmax
            labels_temp = labels_temp.numpy()
            self.y_tilde[indices, :] = labels_temp
        else:
            self.y_prev = cp.deepcopy(self.y_tilde[indices,:]) #Get unnormalized label estimates
            self.y_prev = torch.tensor(self.y_prev).float()
            self.y_prev = self.y_prev.cuda()
            self.y_prev.requires_grad = True
            # obtain label distributions (y_hat)
            y_h = self.softmax(self.y_prev)
            if epoch<self.n_epochs[0]+self.n_epochs[1] or self.use_KL: # During phase 1. If using KL also during phase 2
                lc = self.KLLoss(self.logsoftmax(self.y_prev),self.softmax(outputs))
            else: # During phase 2 use CE if self.use_KL=False
                lc = self.CELoss(self.softmax(outputs),self.softmax(y_h))
            lo = self.CELoss(y_h, labels) # lo is compatibility loss
            le = - torch.mean(torch.mul(self.softmax(outputs), self.logsoftmax(outputs))) # le is entropy loss
        # Compute total loss
        if epoch < self.n_epochs[0]:
            loss = lc
        elif epoch < self.n_epochs[0]+self.n_epochs[1]:
            loss = lc + self.alpha * lo + self.beta * le
            if self.save_losses: self.losses.append([lc.item(),lo.item(),le.item()])
        else:
            loss = lc
        return loss
    
    def update_y_tilde(self, epoch, indices):
        '''
        Call this after the backward pass over the loss
        ''' 
        # If in phase 2, update y estimate
        if epoch >= self.n_epochs[0] and epoch < self.n_epochs[0]+self.n_epochs[1]:
            # update y_tilde by back-propagation
            self.y_tilde[indices]+=-self.gamma*self.y_prev.grad.data.cpu().numpy()
