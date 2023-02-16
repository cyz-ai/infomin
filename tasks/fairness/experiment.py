import numpy as np
import torch
import math, time
import sklearn
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

import utils_os, utils_data
from tasks.fairness.model import FairNet
from mi.renyi import RenyiInfominLayer



## Train function
def train(hyperparam, estimator, x, y, net, DATASET):
    # [A]. some preparation
    dim_z_learnt = hyperparam.dim_learnt
    device = x.device
    SAVE_DIR = 'results/{}/d{}/{}'.format(DATASET, dim_z_learnt, estimator)       
    
    # [B]. learn fair representation z
    net.train().to(device)
    NNMinmaxOptimizer.learn(net, x=x, y=y)
    net.eval()
    utils_os.save_model('{}/model_main{}'.format(SAVE_DIR, dim_z_learnt), net)
    print('\n')

    # [C]. I(Z; Y) estimator training
    print('training network to estimate rho*(Z;Y)...')
    z_learnt, _ = net.encode(x, details=True)
    z_learnt = z_learnt.clone().detach()
    infer_net = net.predict_layer
    infer_net.train()
    loss = infer_net.learn(z_learnt, y)
    print('[val] rho*(Z;Y)=', -loss)   
    utils_os.save_model('{}/model_pred{}'.format(SAVE_DIR, dim_z_learnt), infer_net)
    print('\n')

    # [D]. I(Z; T) estimator training
    print('training network to estimate rho*(Z;T)...')
    z_learnt, z_sensitive = net.encode(x, details=True)                         
    z_learnt, z_sensitive = z_learnt.clone().detach(), z_sensitive.clone().detach() 
    renyi_net = RenyiInfominLayer([net.dim_learnt, 200, net.dim_sensitive], hyperparam)   
    renyi_net.max_iteration = 1000
    renyi_net.debug = True
    renyi_net.to(device)
    loss = renyi_net.learn(z_learnt, z_sensitive)
    print('[val] rho*(Z;T) =', -loss)      
    utils_os.save_model('{}/model_adv{}'.format(SAVE_DIR, dim_z_learnt), renyi_net)
    print('\n')
    

    
    
    
    
## Test function
def test(hyperparam, estimator, x_test, y_test, net, DATASET):
    # [A]. some preparations
    dim_z_learnt = hyperparam.dim_learnt
    device = x_test.device
    SAVE_DIR = 'results/{}/d{}/{}'.format(DATASET, dim_z_learnt, estimator)
    
    # [B]. I(Z; Y) on test set
    z_test, _ = net.encode(x_test, details=True)                                                                      
    net2 = net.predict_layer
    net2.eval().to(device)
    loss = net2.estimate(z_test, y_test)
    print('[test] rho*(Z;Y)=', -loss)

    # [C]. I(Z; T) on test set 
    z_learnt, z_sensitive = net.encode(x_test, details=True)
    z_learnt, z_sensitive = z_learnt.clone().detach(), z_sensitive.clone().detach() 
    renyi_net = RenyiInfominLayer([net.dim_learnt, 200, net.dim_sensitive], hyperparam)         
    utils_os.load_model('{}/model_adv{}'.format(SAVE_DIR, dim_z_learnt), renyi_net)   
    renyi_net.to(device)
    loss = renyi_net.estimate(z_learnt, z_sensitive)
    print('[test] rho*(Z;T)=', -loss) 
    
    
    
    


## Minmax training of neural net    
class NNMinmaxOptimizer(nn.Module):
    
    @staticmethod 
    def learn(net, x, y):    
        # hyperparams 
        T = 500 if not hasattr(net.hyperparams, 'max_iteration') else net.hyperparams.max_iteration
        early_stop = True if not hasattr(net.hyperparams, 'early_stop') else net.hyperparams.early_stop
  
        # divide train & val         
        x_train, y_train, x_val, y_val = utils_data.divide_train_val(x, y)
 
        # optimizer
        optimizer = torch.optim.Adam(net.non_infomin_module(), lr=net.lr, weight_decay=net.wd)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50)
 
        # main loop
        best_val_obj, best_model_state_dict, best_t = -math.inf, None, -1
        for t in range(T):
            # shuffle the data
            idx = torch.randperm(len(x_train)) 
            x_train, y_train = x_train[idx], y_train[idx]

            # max-step
            t0 = time.time()  
            net.train()  
            adv_loss = net.train_infomin_layer(x, y)
            t1 = time.time()
 
            # record the best model
            net.eval()  
            loss_val = -net.objective_func(x_val, y_val)
            if (-loss_val).item() > best_val_obj and t >= 100:
                best_val_obj, best_t, best_model_state_dict = (-loss_val).item(), t, deepcopy(net.state_dict())

            # min-step
            net.train()
            x_chunks, y_chunks = utils_data.get_trunks(x_train, y_train, net.bs)
            for i in range(len(x_chunks)):
                optimizer.zero_grad()            
                loss = -net.objective_func(x_chunks[i], y_chunks[i])
                loss.backward()
                optimizer.step()
               
            # schedule lr
            sched.step(loss_val)

            # report
            if t%(T//10) == 0: 
                print('t=', t, 'loss=', loss.item(), 'loss val=', loss_val.item(), 'adv_loss=', adv_loss, 'time=', t1-t0)

        # early stopping
        if early_stop: net.load_state_dict(best_model_state_dict)
        print('best val loss=', best_val_obj, 't=', t, 'best_t=', best_t, 'early stopping=', early_stop)
        return best_val_obj
    
    
    
    
   
    
    
