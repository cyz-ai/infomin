import torch
import torch.nn as nn
import torch.nn.functional as F

from . import base


class RenyiInfominLayer(base.BaseInfominLayer):
    '''
        sub-network used in infomin, trained by SGD
    '''
    def __init__(self, architecture, hyperparams):
        super().__init__()
        self.main = nn.Sequential(
            *(nn.Linear(architecture[i], architecture[i + 1], bias=True) for i in range(len(architecture) - 2)),
        )
        L = len(self.main)
        self.out = nn.Linear(architecture[-2] * L + 4 * architecture[0], architecture[-1], bias=False)

    def forward(self, z):
        zz = self.features(z)
        out = self.out(zz)
        return out

    def features(self, z):
        z2 = torch.cat([z**0, z, z**2, z**3], dim=1)
        h = z
        H = z2
        for layer in self.main:
            h = layer(h)
            n, d = h.size()
            h = F.leaky_relu(h)
            H = torch.cat((0 * H, h), dim=1)
        return H

    # the neural renyi value
    def objective_func(self, z1, z2):
        return renyi_min_neural(self, z1, z2)

    def learn(self, z1, z2):
        return base.OptimizationHelper.optimize(self, z1, z2)


def renyi_min_neural(adv_layer, z, y, detach=False):
    n, d = z.size()
    z, y = base.standardize_(z), base.standardize_(y)
    y2 = adv_layer(z)
    y2 = base.standardize_(y2, detach=detach)
    yy = (y * y2).mean(dim=0)
    corr = yy.abs()
    mi = corr.mean()
    return mi
