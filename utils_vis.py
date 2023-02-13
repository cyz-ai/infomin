import numpy as np
import matplotlib.pyplot as pyplot
import torch
from sklearn import manifold


def random_pick_data_from_diff_classes(data_class_1, data_class_2):
    N = 1000
    idx1, idx2 = torch.randperm(len(data_class_1)), torch.randperm(len(data_class_2))
    adv_x1, adv_x2 = data_class_1[idx1[:N]].to(device), data_class_2[idx2[:N]].to(device)   
    return adv_x1, adv_x2

def tsne(latent):
    mds = manifold.TSNE(n_components=2, init='pca', random_state=0)
    return mds.fit_transform(latent)

def plot_representation(data_class_1, data_class_2, network, last_layer=True, save_name='tsne.png'):
    # feedforward propagation to get Z content
    network.to(device)
    network.eval()
    with torch.no_grad():
        Z = network.forward(adv_x1, adv_x2)[0]
        classifier_layers = network.content_classifier.main
        L = len(classifier_layers)
        for l in range(L-1):
            if last_layer is True: 
                Z = classifier_layers[l](Z)
    Z = Z.detach().cpu().numpy()
    T =  torch.Tensor([1] * len(adv_x1) + [0] * len(adv_x2)).long().cpu().numpy()
    
    # plot
    embeddings_tsne = tsne(Z)
    N = len(Z)
    plt.figure(figsize=(4, 4))
    plt.scatter(embeddings_tsne[0:int(N/2), 0], embeddings_tsne[0:int(N/2), 1], 5, c='b', alpha = 0.7)
    plt.scatter(embeddings_tsne[int(N/2):, 0], embeddings_tsne[int(N/2):, 1], 5, c='crimson', alpha = 0.4)
    plt.legend(['Domain 1', 'Domain 2'])
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    #plt.show()
    plt.savefig(save_name, dpi=600, bbox_inches = 'tight', pad_inches = 0)