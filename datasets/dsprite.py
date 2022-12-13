import os
from typing import Tuple
from urllib import request

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
from torch.utils.data.sampler import SubsetRandomSampler


class DSpritesDataset(Dataset):
    URL = "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"

    def __init__(self, save_dir='./data', transform=None, target_transform_class=None, filter_scale=None):
        path = os.path.join(save_dir, 'dsprites.npz')
        if not os.path.exists(path):
            request.urlretrieve(self.URL, path)
        self.data = np.load(path, encoding="bytes", allow_pickle=True)

        self.imgs = self.data['imgs']
        self.latents_values = self.data['latents_values']
        self.latents_classes = self.data['latents_classes']
        self.metadata = self.data['metadata'][()]

        if filter_scale is not None:
            selected = (self.latents_classes[:, 2] >= filter_scale[0]) & (self.latents_classes[:, 2] <= filter_scale[1])
            self.imgs = self.imgs[selected]
            self.latents_values = self.latents_values[selected]
            self.latents_classes = self.latents_classes[selected]

        self.transform = transform
        self.target_transform_class = target_transform_class

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        imgs = self.imgs[idx]
        values = self.latents_values[idx]
        classes = self.latents_classes[idx]
        if self.transform:
            imgs = self.transform(imgs)
        if self.target_transform_class:
            classes = self.target_transform_class(classes)
        return imgs, classes


def load_dsprite(
    save_dir='./data',
    downsample_pct: float = 0.5, train_pct: float = 0.8, batch_size: int = 50, img_size: int = 32,
    transform = None, target_transform_class = None,
    label_list: list = None, filter_scale = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if transform is None:
        transform = transforms.Compose([
            torch.FloatTensor,
            lambda x: torch.unsqueeze(x, 0),
            #transforms.Resize(img_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])

    dataset = DSpritesDataset(
        save_dir=save_dir,
        transform=transform, target_transform_class=target_transform_class, filter_scale=filter_scale)
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

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_set, val_set, test_set, train_loader, val_loader, test_loader
