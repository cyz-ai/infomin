import torch
import torch.nn as nn
import torch.nn.functional as F

from . import base, standardize_


def pearson(C, D=None):
    if D is not None:
        z, y = C, D
        m, d = z.size()
        m, d2 = y.size()
        zy = torch.cat((z,y), dim=1)
        zy = standardize_(zy)
        corr = torch.matmul(zy.t(), zy)/(m-1)
        corr = corr[0:d, d:].abs()
        mi = corr.mean()    # <-- can consider max
    else:
        z = C
        m, d = z.size()
        z = standardize_(z)
        corr = torch.matmul(z.t(), z)/(m-1)
        mask = 1 - torch.eye(d).to(z.device)
        corr2 = (corr*mask).abs()
        mi = corr2.mean()   # <-- can consider max
    return mi


def DC(C, D, order=1):
    z, y = D, C
    m, d = z.size()
    z, y = z.contiguous(), y.contiguous()
    A = torch.cdist(z, z, p=2)
    B = torch.cdist(y, y, p=2)
    A_row_sum, A_col_sum = A.sum(dim=0, keepdim=True), A.sum(dim=1, keepdim=True)
    B_row_sum, B_col_sum = B.sum(dim=0, keepdim=True), B.sum(dim=1, keepdim=True)
    a = A - A_row_sum/(m-2) - A_col_sum/(m-2) + A.sum()/((m-1)*(m-2))
    b = B - B_row_sum/(m-2) - B_col_sum/(m-2) + B.sum()/((m-1)*(m-2))
    if order == 1:
        AB, AA, BB = (a*b).sum()/(m*(m-3)), (a*a).sum()/(m*(m-3)), (b*b).sum()/(m*(m-3))
        mi = AB**0.5/(AA**0.5 * BB**0.5)**0.5
    else:
        a, b = a.view(m*m, 1), b.view(m*m, 1)
        c1, c2 = renyi_min(a, b)**0.5, renyi_min(b, a)**0.5
        mi = c1 if c1>=c2 else c2
    return mi


class NonparamInfominLayer(base.BaseInfominLayer):
    ''' sub-network used in infomin, non-parametric '''
    def __init__(self, hyperparams=None):
        super().__init__(hyperparams=hyperparams)
        self.estimator_func = lambda x, y: torch.zeros(1).to(x.device)

    def objective_func(self, z, y):
        return self.estimator_func(z, y)

    def learn(self, z1, z2):
        return torch.zeros(1).to(z1.device)


class DCInfominLayer(NonparamInfominLayer):

    def __init__(self, hyperparams=None):
        super().__init__(hyperparams=hyperparams)
        self.estimator_func = DC


class PearsonInfominLayer(NonparamInfominLayer):

    def __init__(self, hyperparams=None):
        super().__init__(hyperparams=hyperparams)
        self.estimator_func = pearson

