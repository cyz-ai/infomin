import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd as autograd
import numpy as np
import scipy.linalg as linalg
import math
import time
import mi
import utils_data
from mi import base
from copy import deepcopy



class FairNet(nn.Module):
    """ 
        Network that learn fair representation for prediction
    """
    def __init__(self, architecture, dim_y, hyperparams, problem=None):
        super().__init__()
        
        # default hyperparameters
        self.estimator = 'NONE' if not hasattr(hyperparams, 'estimator') else hyperparams.estimator  
        self.bs = 500 if not hasattr(hyperparams, 'bs') else hyperparams.bs 
        self.lr = 1e-4 if not hasattr(hyperparams, 'lr') else hyperparams.lr
        self.wd = 0e-5 if not hasattr(hyperparams, 'wd') else hyperparams.wd
        self.n_neg = 200 if not hasattr(hyperparams, 'n_neg') else hyperparams.n_neg
        self.beta = 0 if not hasattr(hyperparams, 'beta') else hyperparams.beta
        self.non_linearity = torch.sin if not hasattr(hyperparams, 'non_linearity') else hyperparams.non_linearity
        self.hyperparams = hyperparams

        # dimensionality
        self.dim_learnt = architecture[-1] 
        self.dim_sensitive = hyperparams.dim_sensitive

        # network modules
        self.encode_layer1 = EncodeLayer(architecture[0:-1] + [self.dim_learnt], hyperparams)     
        self.encode_layer2 = EncodeLayer(architecture[0:-1] + [self.dim_sensitive], hyperparams)
        self.predict_layer = mi.RenyiInfominLayer([self.dim_learnt, 100, 100, dim_y], hyperparams)
        self.infomin_layer = self.init_infomin_layer(hyperparams)  
        
    def non_infomin_module(self):
        ret = [{'params': self.encode_layer1.parameters()}, 
               {'params': self.encode_layer2.parameters()}, 
               {'params': self.predict_layer.parameters()}]
        return ret
        
    def infomin_module(self):
        ret = [{'params': self.infomin_layer.parameters()}]
        return ret
    
    def train_infomin_layer(self, x, y): 
        z1, z2 = self.encode(x, details=True)
        z1, z2 = z1.clone().detach(), z2.clone().detach()
        return self.infomin_layer.learn(z1, z2)
    
    def init_infomin_layer(self, hyperparams):
        if hyperparams.estimator == 'SLICE':
            return mi.SliceInfominLayer([self.dim_learnt, hyperparams.n_slice, self.dim_sensitive], hyperparams)
        if hyperparams.estimator == 'CLUB':
            return mi.ClubInfominLayer(self.dim_learnt, self.dim_sensitive, hyperparams=hyperparams)
        if hyperparams.estimator == 'RENYI':
            return mi.RenyiInfominLayer([self.dim_learnt, 128, self.dim_sensitive], hyperparams)
        if hyperparams.estimator == 'TC':
            return mi.TCInfominLayer(self.dim_learnt, self.dim_sensitive, hyperparams=hyperparams)
        if hyperparams.estimator == 'DC':
            return mi.DCInfominLayer(hyperparams=hyperparams)
        if hyperparams.estimator == 'PEARSON':
            return mi.PearsonInfominLayer(hyperparams=hyperparams)
        if hyperparams.estimator == 'NONE':
            return mi.PearsonInfominLayer(hyperparams=hyperparams)
        return infomin_layer

    def encode(self, x, details=False):
        z_learnt = self.non_linearity(self.encode_layer1(x))
        z_sensitive = x[:, -1:]
        z = torch.cat((z_learnt,z_sensitive), dim=1) 
        if not details:
            return z
        else:
            return z_learnt, z_sensitive
    
    def objective_func(self, x, y, mode=None):
        z_learnt, z_sensitive = self.encode(x, details=True)
        sufficiency = self.predict_layer.objective_func(z_learnt, y)
        sensitivity = self.infomin_layer.objective_func(z_learnt, z_sensitive)  
        if mode == '+': 
            return sufficiency
        if mode == '-': 
            return sensitivity
        if mode is None:
            return self.beta*sufficiency - sensitivity
        
        
        
        
        
class EncodeLayer(nn.Module):
    '''
        sub-network to compute statistics: z = s(x)
    '''
    def __init__(self, architecture, hyperparams=None):
        super().__init__()
        self.dropout = False if not hasattr(hyperparams, 'dropout') else hyperparams.dropout 
        self.bs = 200 if not hasattr(hyperparams, 'bs') else hyperparams.bs 
        self.lr = 5e-4 if not hasattr(hyperparams, 'lr') else hyperparams.lr
        self.wd = 0e-5 if not hasattr(hyperparams, 'wd') else hyperparams.wd
        
        self.type = 'plain' if not hasattr(hyperparams, 'type') else hyperparams.type 
        self.dropout = False if not hasattr(hyperparams, 'dropout') else hyperparams.dropout 
        self.main = nn.Sequential( 
           *(nn.Linear(architecture[i+1], architecture[i+2], bias=True) for i in range(len(architecture)-3)),
        )      
        self.head = nn.Linear(architecture[0], architecture[1], bias=True)
        self.drop = nn.Dropout(p=0.20)
        self.out = nn.Linear(architecture[-2], architecture[-1], bias=True)
        self.N_layers = len(architecture) - 1
        self.adv_layer = False
                    
    def forward(self, x):
        x = self.head(x)
        for layer in self.main: x = F.leaky_relu(layer(x), 0.2)
        x = self.drop(x) if self.dropout else x
        x = self.out(x)
        return x     
