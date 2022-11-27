import torch
import torch.nn as nn
import torch.nn.functional as F

from . import base


class TCInfominLayer(base.BaseInfominLayer):
    '''
        sub-network used in infomin learning, estimate I(Z; T) by a classifier
    '''
    def __init__(self, dim_x, dim_y, dim_hidden=100, hyperparams=None):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(dim_x + dim_y, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
        )
        self.out = nn.Linear(dim_hidden, 2)

    def forward(self, x, y, others=None):
        h = torch.cat((x,y), dim=1)
        if others is None:
            h = self.main(h)
        else:
            h = self.main(h) + self.cond(others)
        h = F.leaky_relu(h)
        out = self.out(h)
        return out

    def objective_func(self, x, y):
        return -TC(x, y, self)

    def learn(self, x, y):
        return base.OptimizationHelper.optimize(self, x, y)


def TC(z, y, critic):
    m, d = z.size()
    idx_neg1 = torch.randperm(m).cpu().numpy().tolist()
    idx_neg2 = torch.randperm(m).cpu().numpy().tolist()
    f_pos = critic(z, y)
    f_neg = critic(z[idx_neg1], y[idx_neg2])
    ones = torch.ones(m, dtype=torch.long, device=z.device)
    zeros = torch.zeros(m, dtype=torch.long, device=z.device)
    tc = 0.5 * (F.cross_entropy(f_pos, zeros) + F.cross_entropy(f_neg, ones))
    return tc
