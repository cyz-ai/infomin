import torch
import torch.optim as optim

import utils_os


def exp_run(
        train_loaders, test_loaders,
        train, test,
        infomin_batch_provider, model_naming, hyperparams, device='cuda:0', model=None,
        scheduler_func=None):
    best_model_state_dict, best_loss, best_epoch = None, 9999, 0

    optimizer = optim.Adam(model.non_infomin_module(), lr=hyperparams.learning_rate,  betas=(0.5, 0.999))
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
