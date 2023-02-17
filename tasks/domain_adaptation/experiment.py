from itertools import cycle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import mi
import utils_os
from .model import EncoderDA, Classifier


class DomainAdaptation(nn.Module):

    def __init__(self, dim_z_content, dim_z_domain, num_class, num_domain, hyperparams):
        super().__init__()
        self.hyperparams = hyperparams

        self.dim_z_content = dim_z_content
        self.dim_z_domain = dim_z_domain

        self.content_feature = EncoderDA(self.dim_z_content)
        self.domain_feature = EncoderDA(self.dim_z_domain)

        self.content_classifier = Classifier(self.dim_z_content, num_class)        # <-- use to learn z_content
        self.domain_classifier = Classifier(self.dim_z_domain, num_domain)         # <-- use to learn z_domain

        self.infomin_layer = self.init_infomin_layer(hyperparams)

    def init_infomin_layer(self, hyperparams):
        if hyperparams.estimator == 'SLICE':
            return mi.SliceInfominLayer([self.dim_z_content, hyperparams.n_slice, self.dim_z_domain], hyperparams)
        if hyperparams.estimator == 'CLUB':
            return mi.ClubInfominLayer(self.dim_z_content, self.dim_z_domain, hyperparams=hyperparams)
        if hyperparams.estimator == 'RENYI':
            return mi.RenyiInfominLayer([self.dim_z_content, 256, self.dim_z_domain], hyperparams)
        if hyperparams.estimator == 'TC':
            return mi.TCInfominLayer(self.dim_z_content, self.dim_z_domain, hyperparams=hyperparams)
        if hyperparams.estimator == 'DC':
            return mi.DCInfominLayer(hyperparams=hyperparams)
        if hyperparams.estimator == 'PEARSON':
            return mi.PearsonInfominLayer(hyperparams=hyperparams)
        if hyperparams.estimator == 'NONE':
            return mi.NonparametricInfoEstimator(hyperparams=hyperparams)
        return infomin_layer

    def non_infomin_module(self):
        ret = [
            {'params': self.content_classifier.parameters()},
            {'params': self.domain_classifier.parameters()},
            {'params': self.content_feature.parameters()},
            {'params': self.domain_feature.parameters()},
        ]
        return ret

    def train_infomin_layer(self, x1, x2):
        with torch.no_grad():
            z1, z2 = [], []
            x1_chunks, x2_chunks = torch.chunk(x1, 10), torch.chunk(x2, 10)
            for i in range(10):
                z_content, z_domain, y_content_1, y_content_2, y_domain = self.forward(x1_chunks[i], x2_chunks[i])
                z1, z2 = z1 + [z_content.clone().detach()], z2 + [z_domain.clone().detach()]
            z1, z2 = torch.cat(z1, dim=0), torch.cat(z2, dim=0)
        idx = torch.randperm(len(z1))                                 # <--- important to shuffle the data (because we always use the first 80% to train)
        z1, z2 = z1[idx], z2[idx]
        return self.infomin_layer.learn(z1, z2)

    def separate_feature(self, x):
        z_content = self.content_feature(x)
        z_domain = self.domain_feature(x)
        return z_content, z_domain

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=0)
        num_x1 = len(x1)
        z_content, z_domain = self.separate_feature(x)
        z_content_1, z_domain_1 = z_content[:num_x1], z_domain[:num_x1]
        z_content_2, z_domain_2 = z_content[num_x1:], z_domain[num_x1:]

        y_content_1 = self.content_classifier(z_content_1)
        y_content_2 = self.content_classifier(z_content_2)
        y_domain = self.domain_classifier(z_domain)
        return z_content, z_domain, y_content_1, y_content_2, y_domain


def loss_func(model, x1, x2, y1, alpha=1.0, beta=1.0, gamma=1.0):
    z_content, z_domain, y_content_1, y_content_2, y_domain = model(x1, x2)
    t_domain = torch.Tensor([1] * len(x1) + [0] * len(x2)).to(x1.device).long()

    loss_content = nn.CrossEntropyLoss()(y_content_1, y1)
    loss_domain = nn.CrossEntropyLoss()(y_domain, t_domain)

    if model.hyperparams.estimator == 'SLICE':
        redundancy = model.infomin_layer.objective_func(z_content, z_domain)
    else:
        redundancy = model.infomin_layer.objective_func(z_content, z_domain.clone().detach())    # for CLUB and Neural TC, will not converge if we don't detach

    return alpha * loss_content + beta * loss_domain + gamma * redundancy, {
        "loss_content": loss_content,
        "loss_domain": loss_domain,
        "redundancy": redundancy,
    }


def train(epoch, model, optimizer, loaders, infomin_batch_provider, hyperparams, scheduler=None):
    source_loader, target_loader = loaders

    device = hyperparams.device
    model.train()
    batch_idx = 0

    for (x1, y1), (x2, y2) in zip(source_loader, cycle(target_loader)):
        optimizer.zero_grad()

        # prepare data
        x1, x2, y1 = x1.to(device), x2.to(device), y1.to(device)

        # max step
        if batch_idx % hyperparams.update_inner_every_num_epoch == 0:
            infomin_x1, infomin_x2 = infomin_batch_provider()
            infomin_x1, infomin_x2 = infomin_x1.to(device), infomin_x2.to(device)

            infomin_x1, infomin_x2 = torch.cat([infomin_x1, x1], dim=0), torch.cat([infomin_x2, x2], dim=0)

            loss_max_step = model.train_infomin_layer(infomin_x1.to(device), infomin_x2.to(device))
        batch_idx+=1

        # calculate loss & optimise
        loss, _ = loss_func(model, x1, x2, y1, hyperparams.alpha, hyperparams.beta, hyperparams.gamma)
        loss.backward()
        # clip gradient
        if hyperparams.grad_clip is not None: nn.utils.clip_grad_value_(model.parameters(), hyperparams.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

    # schedule lr
    if scheduler:
        if 'ReduceLROnPlateau' in str(scheduler):
            scheduler.step(loss)
        else:
            scheduler.step()

    return loss.item()


def test(epoch, model, loaders, hyperparams):
    source_loader, target_loader = loaders

    device = hyperparams.device
    model.eval()

    correct_d1, correct_d2 = 0, 0
    num_d1, num_d2 = 0, 0
    correct_domain = 0
    num_domain = 0

    correct_domain2 = 0
    num_domain2 = 0

    losses, losses1, losses2, losses3 = [], [], [], []
    with torch.no_grad():
        for (x1, y1), (x2, y2) in zip(source_loader, cycle(target_loader)):
            # FP in model
            x1, y1, x2, y2 = x1.to(device), y1.to(device), x2.to(device), y2.to(device)
            z_content, z_domain, y_content_1, y_content_2, y_domain = model(x1, x2)

            # calculate content accuracy
            correct_d1 += torch.sum(y_content_1.argmax(dim=1) == y1)
            correct_d2 += torch.sum(y_content_2.argmax(dim=1) == y2)
            num_d1 += len(y_content_1)
            num_d2 += len(y_content_2)

            # calculate domain accuracy (z_domain only)
            target = torch.Tensor([1] * len(x1) + [0] * len(x2)).to(x1.device)
            correct_domain += torch.sum(y_domain.argmax(dim=1) == target)
            num_domain += len(y_domain)

            # loss statistics
            loss, info = loss_func(
                model, x1, x2, y1,
                hyperparams.alpha, hyperparams.beta, hyperparams.gamma
            )
            losses.append(loss.item())
            losses1.append(info['loss_content'].item())
            losses2.append(info['loss_domain'].item())
            losses3.append(info['redundancy'].item())

    # record
    acc_d1 = 100. * correct_d1 / num_d1
    acc_d2 = 100. * correct_d2 / num_d2
    acc_domain = 100. * correct_domain / num_domain
    loss = np.array(losses).mean()
    loss_content = np.array(losses1).mean()
    loss_domain = np.array(losses2).mean()
    redundancy = np.array(losses3).mean()

    print(f'epoch: {epoch}, loss: {loss}, loss_content: {loss_content},' \
          f' loss_domain: {loss_domain} redundancy: {redundancy}', \
          f' acc_d1: {acc_d1} acc_d2: {acc_d2} acc_domain: {acc_domain}\n')

    log = {
        'redundancy': redundancy,
        'loss_content': loss_content,
        'loss_domain': loss_domain,
        'acc_d1': acc_d1,
        'acc_d2': acc_d2,
        'acc_domain': acc_domain,
    }

    return loss, log


def exp_run(
        train_loaders, test_loaders,
        train, test,
        infomin_batch_provider, model_naming, hyperparams, device='cuda:0', model=None,
        scheduler_func=None):
    best_model_state_dict, best_loss, best_epoch = None, 9999, 0

    optimizer = torch.optim.Adam(model.non_infomin_module(), lr=hyperparams.learning_rate,  betas=(0.5, 0.999))
    scheduler = None if not scheduler_func else scheduler_func(optimizer)

    for epoch in range(0, hyperparams.n_epochs + 1):
        # train
        train(epoch, model, optimizer, train_loaders, infomin_batch_provider, hyperparams, scheduler=scheduler)

        # test
        loss, _ = test(epoch, model, test_loaders, hyperparams)

        # early stopping
        if loss < best_loss:
            best_loss, best_epoch = loss, epoch
            best_model_state_dict = model.state_dict()

        # save model
        if epoch % 20 == 0:
            utils_os.save_model(model_naming(hyperparams, epoch), model)

    # load the best model
    if getattr(hyperparams, 'early_stopping', False): model.load_state_dict(best_model_state_dict)
    utils_os.save_model(model_naming(hyperparams), model)
    return model