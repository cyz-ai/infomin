import os
from typing import Tuple
from urllib import request

import numpy as np
import cv2
from scipy import io as sio
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
from torch.utils.data.sampler import SubsetRandomSampler


class TorontoFaceDataset(Dataset):
    """Original implementation is:
        https://github.com/nouiz/lisa_emotiw/blob/master/emotiw/common/datasets/faces/tfd.py.
    """

    URL_DATA = "http://www.cs.toronto.edu/~jsusskin/TFD/TFD_48x48.mat"
    URL_INFO = "http://www.cs.toronto.edu/~jsusskin/TFD/TFD_info.mat"

    def __init__(self, save_dir='./data', transform=None, target_transform=None):
        path_data = os.path.join(save_dir, 'TFD_48x48.mat')
        path_info = os.path.join(save_dir, 'TFD_info.mat')
        if not os.path.exists(path_data):
            request.urlretrieve(self.URL_DATA, path_data)
            request.urlretrieve(self.URL_INFO, path_info)

        tfd_48x48 = sio.loadmat(path_data)
        self.imgs = tfd_48x48["images"]
        self.labels_ex = tfd_48x48['labs_ex']
        self.labels_id = tfd_48x48['labs_id']
        self.folds = tfd_48x48['folds']
        self.imgs = self.imgs[self.labels_ex[:, 0] != -1]
        self.labels_id = self.labels_id[self.labels_ex[:, 0] != -1]
        self.folds = self.folds[self.labels_ex[:, 0] != -1]
        self.labels_ex = self.labels_ex[self.labels_ex[:, 0] != -1]

        # Load mapping file to original dataset
        self.mapping = sio.loadmat(path_info)
        self.mapping = [x[0].encode('ascii', 'ignore') for x in self.mapping["tfd_info"]["imfiles"][0, 0].flatten()]
        #print(self.mapping[:100])

        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        imgs = self.imgs[idx]
        labels_ex = self.labels_ex[idx]
        labels_id = self.labels_id[idx]
        folds = self.folds[idx]
        if self.transform:
            imgs = self.transform(imgs)
        return imgs, labels_ex, labels_id, folds


def load_tfd(
    downsample_pct: float = 0.5, train_pct: float = 0.8, batch_size: int = 50, img_size: int = 32, label_list: list = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose([
        cv2.equalizeHist,
        torch.FloatTensor,
        lambda x: torch.unsqueeze(x, 0),
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.NEAREST),
        lambda x: x / 256,
    ])

    dataset = TorontoFaceDataset(transform=transform)
    N = len(dataset)
    n = int(N*downsample_pct)
    n_train = int(n*0.80*train_pct)
    n_val = int(n*0.80*(1-train_pct))
    n_test = int(n*0.20)
    n_other = N-(n_train+n_val+n_test)

    train_set, val_set, test_set, _ = torch.utils.data.random_split(
        dataset,
        lengths=[n_train, n_val, n_test, n_other],
    )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_set, val_set, test_set, train_loader, val_loader, test_loader
