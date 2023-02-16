import torch
import torch.nn as nn
import torch.nn.functional as F
import utils_data

from . import base


def estimate(C, D=None):
    if D is not None:
        z, y = C, D
        m, d = z.size()
        m, d2 = y.size()
        zy = torch.cat((z,y), dim=1)
        zy = utils_data.standardize_(zy)
        corr = torch.matmul(zy.t(), zy)/(m-1)
        corr = corr[0:d, d:].abs()
        mi = corr.mean()    # <-- can consider max
    else:
        z = C
        m, d = z.size()
        z = utils_data.standardize_(z)
        corr = torch.matmul(z.t(), z)/(m-1)
        mask = 1 - torch.eye(d).to(z.device)
        corr2 = (corr*mask).abs()
        mi = corr2.mean()   # <-- can consider max
    return mi


class PearsonInfominLayer(base.NonparametricInfoEstimator):

    def __init__(self, hyperparams=None):
        super().__init__(hyperparams=hyperparams)
        self.estimator_func = estimate

