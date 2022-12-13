import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import scipy.linalg as linalg
import math
import time
from copy import deepcopy


all = ['standardize_', 'jitter_', 'BaseInfominLayer', 'OptimizationHelper']


def standardize_(x, detach=False):
    if detach:
        x_mu, x_std = x.mean(dim=0, keepdim=True).clone().detach(), x.std(dim=0, keepdim=True).clone().detach()
    else:
        x_mu, x_std = x.mean(dim=0, keepdim=True), x.std(dim=0, keepdim=True)
    return (x-x_mu)/x_std


def jitter_(x):
    noise = torch.randn(x.size()).to(x.device)*1e-6
    return x + noise


class BaseInfominLayer(nn.Module):
    ''' sub-network used in infomin, trained by SGD '''

    def __init__(self, hyperparams={}):
        super().__init__()
        self.lr = hyperparams.get('inner_lr', 5e-4)
        self.max_iteration = hyperparams.get('inner_epochs', 2000)
        self.bs = hyperparams.get('inner_batch_size', 1024)
        self.wd = hyperparams.get('inner_weight_decay', 0e-5)

    def forward(self, x):
        raise NotImplemented

    def objective_func(self, x, y):
        """The objective function for inner steps, assuming it maximizes MI."""
        raise NotImplemented

    def learn(self, x, y):
        raise NotImplemented

    def estimate(self, x, y):
        return OptimizationHelper.estimate(self, x, y)

    def test(self, x, y):
        return OptimizationHelper.test(self, x, y)


class OptimizationHelper():
    DEBUG = False
    T_NO_IMPROVE_THRESHOLD = 800

    @staticmethod
    def divide_train_val(x, y):
        n = len(x)
        n_train = int(0.80*n)
        x_train, y_train = x[0:n_train], y[0:n_train]
        x_val, y_val = x[n_train:n], y[n_train:n]
        return  x_train, y_train, x_val, y_val

    @staticmethod
    def get_trunks(x, y, bs):
        n_batch = int(len(x)/bs) if len(x) > bs else 1
        x_chunks, y_chunks = torch.chunk(x, n_batch), torch.chunk(y, n_batch)
        return x_chunks, y_chunks

    @classmethod
    def optimize(cls, model, x, y):
        # hyperparams
        assert hasattr(model, 'max_iteration'), 'missing hyperparameter max_iteration'
        assert hasattr(model, 'bs'), 'missing hyperparameter bs'
        assert hasattr(model, 'lr'), 'missing hyperparameter lr'
        assert hasattr(model, 'wd'), 'missing hyperparameter wd'
        T = model.max_iteration

        # divide train & val
        n = len(x)
        x_train, y_train, x_val, y_val = cls.divide_train_val(x, y)
        bs = min(model.bs, len(x_train))
        model.device = x.device

        # learn in loops
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=model.lr, weight_decay=model.wd)
        n_batch, n_val_batch = len(x_train) // bs if len(x_train) > bs else 1, len(x_val) // bs if len(x_val) > bs else 1
        best_val_loss, best_model_state_dict, no_improvement = math.inf, model.state_dict(), 0

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

            # early stopping if val loss does not improve after some epochs
            model.eval()
            loss_val = torch.zeros(1, device=x.device)
            for j in range(len(x_v_chunks)):
                loss_val += -model.objective_func(x_v_chunks[j], y_v_chunks[j]) / len(x_v_chunks)

            improved = loss_val.item() < best_val_loss
            no_improvement = 0 if improved else no_improvement + 1
            best_val_loss = loss_val.item() if improved else best_val_loss
            best_model_state_dict = deepcopy(model.state_dict()) if improved else best_model_state_dict

            if no_improvement >= cls.T_NO_IMPROVE_THRESHOLD: break

            # report
            if cls.DEBUG and t % (T // 10) == 0:
                print('finished: t=', t, 'loss=', loss.item(), 'loss val=', loss_val.item(), best_val_loss)

        # return the best snapshot in the history
        model.load_state_dict(best_model_state_dict)
        return best_val_loss

    @staticmethod
    def test(model, x, y):
        x_train, y_train, x_val, y_val = OptimizationMixin.divide_train_val(x, y)
        N, correct = 0.0, 0.0
        with torch.no_grad():
            _y = model.forward(x_val)
            correct += torch.sum(_y.argmax(dim=1) == y_val.argmax(dim=1))
            N += len(_y)
        return correct / N

    @staticmethod
    def estimate(model, x, y):
        x_train, y_train, x_val, y_val = base.MaxStepHelper.divide_train_val(x, y)
        x_chunks, y_chunks = base.MaxStepHelper.get_trunks(x_val, y_val, model.bs)
        model.eval()
        loss = torch.zeros(1, device=x.device)
        for j in range(len(x_chunks)):
            loss += -model.objective_func(x_chunks[j], y_chunks[j]) / len(x_chunks)
        return loss.item()
