import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import EncoderAE, Decoder
import mi


class Autoencoder(nn.Module):

    def __init__(self, dim_learnt, dim_expert, hyperparams, num_color_channel=3):
        super().__init__()
        self.hyperparams = hyperparams
        self.dim_learnt = dim_learnt
        self.dim_expert = dim_expert
        # new feature
        self.new_feature = EncoderAE(self.dim_learnt, num_color_channel=num_color_channel)
        # expert/label feature
        self.label_feature = EncoderAE(self.dim_expert, num_color_channel=num_color_channel)
        self.dec = Decoder(self.dim_learnt + self.dim_expert, num_color_channel=num_color_channel)
        self.infomin_layer = self.init_infomin_layer(hyperparams)
        print('dim learnt', self.dim_learnt, 'dim expert', self.dim_expert)

    def init_infomin_layer(self, hyperparams):
        if hyperparams.estimator == 'SLICE':
            return mi.SliceInfominLayer([self.dim_learnt, hyperparams.n_slice, self.dim_expert], hyperparams)
        if hyperparams.estimator == 'CLUB':
            return mi.ClubInfominLayer(self.dim_learnt, self.dim_expert, hyperparams=hyperparams)
        if hyperparams.estimator == 'RENYI':
            return mi.RenyiInfominLayer([self.dim_learnt, 256, self.dim_expert], hyperparams)
        if hyperparams.estimator == 'TC':
            return mi.TCInfominLayer(self.dim_learnt, self.dim_expert, hyperparams=hyperparams)
        if hyperparams.estimator == 'DC':
            return mi.DCInfominLayer(hyperparams=hyperparams)
        if hyperparams.estimator == 'PEARSON':
            return mi.PearsonInfominLayer(hyperparams=hyperparams)
        if hyperparams.estimator == 'NONE':
            return mi.NonparamInfominLayer(hyperparams=hyperparams)
        raise ValueError('non-supported mi proxy/estimator')

    def non_infomin_module(self):
        ret = [{'params': self.new_feature.parameters()},
               {'params': self.label_feature.parameters()},
               {'params': self.dec.parameters()}]
        return ret

    def train_infomin_layer(self, x, t):
        with torch.no_grad():
            z1 = self.new_feature(x)
            z2 = t.float()
            z1, z2 = z1.clone().detach(), z2.clone().detach()
        return self.infomin_layer.learn(z1, z2)

    def separate_feature(self, x):
        z = self.new_feature(x)
        y = self.label_feature(x)
        return z, y

    def decode(self, z, y):
        zy = torch.cat((z, y), dim=1)
        x_recon = self.dec(zy)
        return x_recon

    def forward(self, x, y):
        zz, _y = self.separate_feature(x)
        x_recon = self.decode(zz, y)
        _y_sm = _y.softmax(dim=1)
        return x_recon, zz, _y_sm


def loss_func(model, x, y, alpha=1.0, beta=1.0, gamma=1.0):
    output, zz, _y = model(x, y)
    zz = model.new_feature(x)
    recon_x, x = output, x
    mse = F.mse_loss(recon_x, x, reduction='mean')
    label_cost = nn.CrossEntropyLoss()(_y, y.argmax(dim=1))
    redundancy = model.infomin_layer.objective_func(zz, y)
    loss = alpha * mse + beta * label_cost + gamma * redundancy
    return loss, {
        'mse': mse,
        'label_cost': label_cost,
        'redundancy': redundancy,
    }


def train(epoch, model, optimizer, loaders, infomin_batch_provider, hyperparams, scheduler=None):
    train_loader, = loaders
    device = hyperparams.device
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        # prepare data
        x, y = data.to(device), target.to(device)
        # max step
        if batch_idx % hyperparams.update_inner_every_num_epoch == 0:
            xx, yy = infomin_batch_provider()
            loss_max_step = model.train_infomin_layer(xx.to(device), yy.to(device))
        # calculate loss function
        loss, _ = loss_func(model, x, y, hyperparams.alpha, hyperparams.beta, hyperparams.gamma)
        loss.backward()
        # clip gradient
        if hyperparams.grad_clip is not None: nn.utils.clip_grad_value_(model.parameters(), hyperparams.grad_clip)
        optimizer.step()
        optimizer.zero_grad()
    # schedule lr
    if scheduler:
        scheduler.step()
    return loss.item()


def test(epoch, model, loaders, hyperparams):
    test_loader, = loaders

    device = hyperparams.device
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        losses = []
        mse = []
        label_cost = []
        redundancy = []
        N = 0
        for data, target in test_loader:
            # FP in model
            x, y = data.to(device), target.to(device)
            recon, zz, _y = model(x, y)

            # calculate acc (don't forget to do standarization)
            correct += torch.sum(_y.argmax(dim=1) == y.argmax(dim=1))
            N += len(_y)

            # calculate sufficiency and redundancy
            loss, info = loss_func(model, x, y, hyperparams.alpha, hyperparams.beta, hyperparams.gamma)
            losses.append(loss.item())
            mse.append(info['mse'].item())
            label_cost.append(info['label_cost'].item())
            redundancy.append(info['redundancy'].item())

        # record
        acc = 100. * correct / N
        loss = np.array(losses).mean()
        mse = np.array(mse).mean()
        label_cost = np.array(label_cost).mean()
        redundancy = np.array(redundancy).mean()
        log = {}
        log['recon'] = recon
        log['x'] = x
        log['acc'] = acc
        log['mse'] = mse
        log['label_cost'] = label_cost
        log['redundancy'] = redundancy
    print(f'epoch: {epoch}, loss: {loss.item()}, mse: {mse}, acc: {acc}, redundancy: {redundancy}')
    return loss, log
