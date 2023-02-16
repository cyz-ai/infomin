import numpy as np
import matplotlib.pyplot as pyplot
import torch


''' 
    useful data processing operations
'''


def standardize_(x, detach=False):
    if detach:
        x_mu, x_std = x.mean(dim=0, keepdim=True).clone().detach(), x.std(dim=0, keepdim=True).clone().detach()
    else:
        x_mu, x_std = x.mean(dim=0, keepdim=True), x.std(dim=0, keepdim=True)
    return (x-x_mu)/x_std


def jitter_(x):
    noise = torch.randn(x.size()).to(x.device)*1e-6
    return x + noise


def divide_train_val(x, y, ratio=0.80):
    n = len(x)
    n_train = int(ratio*n)
    x_train, y_train = x[0:n_train], y[0:n_train]
    x_val, y_val = x[n_train:n], y[n_train:n]
    return  x_train, y_train, x_val, y_val


def get_trunks(x, y, bs):
    n_batch = int(len(x)/bs) if len(x) > bs else 1
    x_chunks, y_chunks = torch.chunk(x, n_batch), torch.chunk(y, n_batch)
    return x_chunks, y_chunks