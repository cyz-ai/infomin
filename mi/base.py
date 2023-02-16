import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import scipy.linalg as linalg
import math
import time
import utils_data
from copy import deepcopy


all = ['ParametricInfoEstimator', 'NonparametricInfoEstimator']



class ParametricInfoEstimator(nn.Module):
    ''' sub-network used in infomin, trained by SGD '''

    def __init__(self, hyperparams={}):
        super().__init__()
        self.lr = hyperparams.get('inner_lr', 5e-4)
        self.max_iteration = hyperparams.get('inner_epochs', 1000)
        self.bs = hyperparams.get('inner_batch_size', 1024)
        self.wd = hyperparams.get('inner_weight_decay', 0e-5)
        self.debug = False

    def forward(self, x):
        raise NotImplemented

    def objective_func(self, x, y):
        """The objective function for inner steps, assuming it maximizes MI."""
        raise NotImplemented

    def learn(self, x, y):
        T_NO_IMPROVE_THRESHOLD = 500
        model = self
        
        # hyperparams
        assert hasattr(model, 'max_iteration'), 'missing hyperparameter max_iteration'
        assert hasattr(model, 'bs'), 'missing hyperparameter bs'
        assert hasattr(model, 'lr'), 'missing hyperparameter lr'
        assert hasattr(model, 'wd'), 'missing hyperparameter wd'
        T = model.max_iteration
        if model.debug: print('T=', T)
        
        # divide train & val
        n = len(x)
        x_train, y_train, x_val, y_val = utils_data.divide_train_val(x, y)
        bs = min(model.bs, len(x_train))
        model.device = x.device

        # learn in loops
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=model.lr, weight_decay=model.wd)
        n_batch, n_val_batch = len(x_train) // bs if len(x_train) > bs else 1, len(x_val) // bs if len(x_val) > bs else 1
        best_val_loss, best_model_state_dict, no_improvement = math.inf, model.state_dict(), 0

        # loop 
        for t in range(T):
            # shuffle the batch
            idx = torch.randperm(len(x_train))
            x_train, y_train = x_train[idx], y_train[idx]
            x_chunks, y_chunks = torch.chunk(x_train, n_batch), torch.chunk(y_train, n_batch)
            x_v_chunks, y_v_chunks = torch.chunk(x_val, n_val_batch), torch.chunk(y_val, n_val_batch)

            # gradient descend
            model.train()
            for i in range(len(x_chunks)):
                optimizer.zero_grad()
                loss = -model.objective_func(x_chunks[i], y_chunks[i])
                if t > 0:
                    loss.backward()
                    optimizer.step()

            # early stopping
            model.eval()
            loss_val = torch.zeros(1, device=x.device)
            for j in range(len(x_v_chunks)):
                loss_val += -model.objective_func(x_v_chunks[j], y_v_chunks[j]) / len(x_v_chunks)
            improved = loss_val.item() < best_val_loss
            if improved:
                no_improvement = 0
                best_val_loss = loss_val.item()
                best_model_state_dict = deepcopy(model.state_dict())
            else:
                no_improvement += 1
            if no_improvement >= T_NO_IMPROVE_THRESHOLD: break

            # report
            if model.debug and t % (T // 10) == 0:
                print('finished: t=', t, 'loss=', loss.item(), 'loss val=', loss_val.item(), best_val_loss)

        # return the best snapshot in the history
        model.load_state_dict(best_model_state_dict)
        return best_val_loss

    def estimate(self, x, y):
        model = self
        x_train, y_train, x_val, y_val = utils_data.divide_train_val(x, y)
        x_chunks, y_chunks = utils_data.get_trunks(x_val, y_val, model.bs)
        model.eval()
        loss = torch.zeros(1, device=x.device)
        for j in range(len(x_chunks)):
            loss += -model.objective_func(x_chunks[j], y_chunks[j]) / len(x_chunks)
        return loss.item()



    
class NonparametricInfoEstimator(nn.Module):
    ''' sub-network used to estimate I(x, y), non-parametric '''
    def __init__(self, hyperparams=None):
        super().__init__()
        self.bs = hyperparams.get('inner_batch_size', 256)
        self.estimator_func = lambda x, y: torch.zeros(1).to(x.device)

    def objective_func(self, x, y):
        return self.estimator_func(x, y)

    def learn(self, x, y):
        return torch.zeros(1).to(x.device)       # <--- there is nothing to learn in non-parametric method
    
    def estimate(self, x, y):
        model = self
        x_train, y_train, x_val, y_val = utils_data.divide_train_val(x, y)
        x_chunks, y_chunks = utils_data.get_trunks(x_val, y_val, model.bs)
        model.eval()
        loss = torch.zeros(1, device=x.device)
        for j in range(len(x_chunks)):
            loss += -model.objective_func(x_chunks[j], y_chunks[j]) / len(x_chunks)
        return loss.item()
    
    
    
    
    
# class OptimizationHelper():
#     DEBUG = False
#     T_NO_IMPROVE_THRESHOLD = 800

#     @classmethod
#     def optimize(cls, self, x, y):

#         # hyperparams
#         assert hasattr(model, 'max_iteration'), 'missing hyperparameter max_iteration'
#         assert hasattr(model, 'bs'), 'missing hyperparameter bs'
#         assert hasattr(model, 'lr'), 'missing hyperparameter lr'
#         assert hasattr(model, 'wd'), 'missing hyperparameter wd'
#         T = model.max_iteration
        
#         # divide train & val
#         n = len(x)
#         x_train, y_train, x_val, y_val = utils_data.divide_train_val(x, y)
#         bs = min(model.bs, len(x_train))
#         model.device = x.device

#         # learn in loops
#         optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=model.lr, weight_decay=model.wd)
#         n_batch, n_val_batch = len(x_train) // bs if len(x_train) > bs else 1, len(x_val) // bs if len(x_val) > bs else 1
#         best_val_loss, best_model_state_dict, no_improvement = math.inf, model.state_dict(), 0

#         for t in range(T):
#             # shuffle the batch
#             idx = torch.randperm(len(x_train))
#             x_train, y_train = x_train[idx], y_train[idx]
#             x_chunks, y_chunks = torch.chunk(x_train, n_batch), torch.chunk(y_train, n_batch)
#             x_v_chunks, y_v_chunks = torch.chunk(x_val, n_val_batch), torch.chunk(y_val, n_val_batch)

#             # gradient descend
#             model.train()
#             for i in range(len(x_chunks)):
#                 optimizer.zero_grad()
#                 loss = -model.objective_func(x_chunks[i], y_chunks[i])
#                 if t > 0:
#                     loss.backward()
#                     optimizer.step()

#             # early stopping if val loss does not improve after some epochs
#             model.eval()
#             loss_val = torch.zeros(1, device=x.device)
#             for j in range(len(x_v_chunks)):
#                 loss_val += -model.objective_func(x_v_chunks[j], y_v_chunks[j]) / len(x_v_chunks)

#             improved = loss_val.item() < best_val_loss
#             no_improvement = 0 if improved else no_improvement + 1
#             best_val_loss = loss_val.item() if improved else best_val_loss
#             best_model_state_dict = deepcopy(model.state_dict()) if improved else best_model_state_dict

#             if no_improvement >= cls.T_NO_IMPROVE_THRESHOLD: break

#             # report
#             if cls.DEBUG and t % (T // 10) == 0:
#                 print('finished: t=', t, 'loss=', loss.item(), 'loss val=', loss_val.item(), best_val_loss)

#         # return the best snapshot in the history
#         model.load_state_dict(best_model_state_dict)
#         return best_val_loss



    # def test(self, x, y):
    #     model = self
    #     x_train, y_train, x_val, y_val = utils_data.divide_train_val(x, y)
    #     N, correct = 0.0, 0.0
    #     with torch.no_grad():
    #         _y = model.forward(x_val)
    #         correct += torch.sum(_y.argmax(dim=1) == y_val.argmax(dim=1))
    #         N += len(_y)
    #     return correct / N

