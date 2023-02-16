import torch
import torch.nn as nn
import torch.nn.functional as F
import utils_data

from . import base


class RenyiInfominLayer(base.ParametricInfoEstimator):
    '''
        sub-network used in infomin, trained by SGD
    '''
    def __init__(self, architecture, hyperparams):
        super().__init__(hyperparams=hyperparams)
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
        return estimate(self, z1, z2)

    def learn(self, z1, z2):
        return super().learn(z1, z2)


def estimate(network, z, y, detach=False):
    n, d = z.size()
    z, y = utils_data.standardize_(z), utils_data.standardize_(y)
    y2 = network(z)
    y2 = utils_data.standardize_(y2, detach=detach)
    yy = (y * y2).mean(dim=0)
    corr = yy.abs()
    mi = corr.mean()
    return mi
