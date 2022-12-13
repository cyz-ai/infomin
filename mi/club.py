import torch
import torch.nn as nn
import torch.nn.functional as F

from . import base


class ClubInfominLayer(base.BaseInfominLayer):
    """ sub-network used in infomin, trained by SGD """
    def __init__(self, dim_z, dim_y, hidden_size=200, hyperparams={}):
        super().__init__(hyperparams=hyperparams)
        self.club = CLUB(dim_z, dim_y, hidden_size)
        self.mode = 'eval'

    def forward(self, x, y):
        return self.club(x, y)

    def objective_func(self, x, y):
        if self.mode == 'learn':                  # <-- max step, log p(y|x)
            return -self.club.learning_loss(x, y)
        if self.mode == 'eval':                   # <-- min step, use bound
            return self.club(x, y)

    def learn(self, x, y):
        self.to(x.device)
        self.mode = 'learn'
        ret = base.OptimizationHelper.optimize(self, x, y)
        self.mode = 'eval'
        return ret


class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    """This class provides the CLUB estimation to I(X,Y)
    Method:
        forward() :      provides the estimation with input samples
        loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
    Arguments:
        x_dim, y_dim :         the dimensions of samples from X, Y respectively
        hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
        x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim]
    """
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim

        # p_mu outputs mean of q(Y|X)
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden_size // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size // 2, y_dim)
        )

        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden_size // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size // 2, y_dim),
            nn.Tanh()
        )

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        # log of conditional probability of positive sample pairs
        positive = -(mu - y_samples) ** 2 / 2. / logvar.exp()

        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = -((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2. / logvar.exp()

        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

    def log_likelihood(self, x_samples, y_samples): # unnormalized loglikelihood
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return -self.log_likelihood(x_samples, y_samples)
