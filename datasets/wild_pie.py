import os
from typing import Tuple
from urllib import request
import glob
import re

from PIL import Image
import numpy as np
from scipy import io as sio
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
from torch.utils.data.sampler import SubsetRandomSampler


class PieDataset(Dataset):
    """Original source:
        https://github.com/bluer555/CR-GAN/blob/master/README.md
    """

    def __init__(self, save_dir='./data/pie', transform=None, target_transform=None):
        self.img_names = glob.glob(f'{save_dir}/*/*/*.png')
        self.labels = [int(
            re.search(r'\d+_\d+_(\d+)_\d+_\d+_crop.*\.png', s).group(1)
        ) for s in self.img_names]

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_file_path = self.img_names[idx]
        label = self.labels[idx]
        with open(img_file_path, "rb") as f:
            img = Image.open(f)
            img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label


def load_pie(
    downsample_pct: float = 0.5, train_pct: float = 0.8, batch_size: int = 50, img_size: int = 32, label_list: list = None,
    transform = None, target_transform = None,
    save_dir = './data/pie'
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ])
    if target_transform is None:
        target_transform = transforms.Compose([
            lambda x: x - 1
        ])

    dataset = PieDataset(save_dir=save_dir, transform=transform, target_transform=target_transform)
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

