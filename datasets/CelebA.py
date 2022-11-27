import torch.utils.data as data
from PIL import Image
from typing import Dict, Tuple
import collections
import os
import torch
import torchvision 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
from random import shuffle


CELEBA_ROOT = 'data/CelebA'
attr_list = [1, 3, 5, 7, 21]

class CelebA_SELECT(torchvision.datasets.ImageFolder):
    
    def __init__(self, transform=None, attr_list=attr_list):
        CelebA_Image_root = os.path.join(CELEBA_ROOT, 'Img')
        super().__init__(root=CelebA_Image_root, transform=transform)
        self.labels = self.get_labels(attr_list)
        self.splits = self.get_split()
                
    def get_labels(self, attr_list=attr_list):
        CelebA_Attr_file = os.path.join(CELEBA_ROOT, 'Anno/list_attr_celeba.txt')
        labels = []
        with open(CelebA_Attr_file, "r") as Attr_file:
            Attr_info = Attr_file.readlines()
            column_names = {val: idx for idx, val in enumerate(Attr_info[1].split(), 1)}
            Attr_info = Attr_info[2:]
            for line in Attr_info:
                info = line.split()
                id = int(info[0].split('.')[0])
                label = np.zeros(len(attr_list))
                for i in range(len(attr_list)):
                    attr = attr_list[i]
                    if isinstance(attr, str):
                        attr = column_names[attr]
                    label[i] = (int(info[attr])+1)/2
                labels.append(label)
        return np.array(labels)

    def __getitem__(self, index):
        img = super().__getitem__(index)[0]
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.labels)

    def get_split(self):
        split_file = os.path.join(CELEBA_ROOT, 'Anno/list_eval_partition.txt')
        partitions = []
        with open(split_file, "r") as file_:
            info = file_.readlines()
            for line in info:
                info = line.strip().split()
                partitions.append(int(info[-1]))
        return np.array(partitions)


def load_celeba(
    downsample_pct: float = 0.5, train_pct: float = 0.8, batch_size: int = 50, img_size: int = 32, label_list: list = None
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose([
                               transforms.Resize(img_size),
                               transforms.CenterCrop(img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = CelebA_SELECT(transform=transform, attr_list=label_list)
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

    return train_loader, val_loader, test_loader


class CelebADS(torch.utils.data.Dataset):
    SPLIT_MAP = {
        'train': 0,
        'val': 1,
        'test': 2,
    }

    def __init__(self, transform=None, target_transform=None, attr_list=attr_list, split='train', downsample_pct=1.0):
        self.image_folder_path = os.path.join(CELEBA_ROOT, 'Img/img_align_celeba')
        self.attr_file_path = os.path.join(CELEBA_ROOT, 'Anno/list_attr_celeba.txt')
        self.split_file_path = os.path.join(CELEBA_ROOT, 'Anno/list_eval_partition.txt')
        self.identity_file_path = os.path.join(CELEBA_ROOT, 'Anno/identity_CelebA.txt')

        self.transform = transform
        self.target_transform = target_transform
        self.attr_list = attr_list
        self.split_type = split

        self.splits = self.load_split()
        self.labels = self.load_labels(attr_list)
        self.img_names = self.load_image_names()
        self.identities = self.load_identities()

        self.labels = self.labels[:int(len(self.labels) * downsample_pct)]
        self.img_names = self.img_names[:int(len(self.img_names) * downsample_pct)]
        self.identities = self.identities[:int(len(self.identities) * downsample_pct)]

    def load_split(self):
        partitions = []
        with open(self.split_file_path, "r") as file_:
            info = file_.readlines()
            for line in info:
                info = line.strip().split()
                partitions.append((info[0], info[1]))
        return np.array(partitions)

    def get_target_split_imgs(self, target):
        return self.splits[self.splits[:, 1] == str(self.SPLIT_MAP[target])][:, 0]

    def load_labels(self, attr_list=attr_list):
        labels = {}
        with open(self.attr_file_path, "r") as attr_file:
            attr_info = attr_file.readlines()
            column_names = {val: idx for idx, val in enumerate(attr_info[1].split(), 1)}
            attr_info = attr_info[2:]
            for line in attr_info:
                info = line.split()
                img_id = info[0]
                label = np.zeros(len(attr_list))
                for i in range(len(attr_list)):
                    attr = attr_list[i]
                    if isinstance(attr, str):
                        attr = column_names[attr]
                    label[i] = (int(info[attr])+1)/2
                labels[img_id] = label
        labels_list = []
        for target_img in self.get_target_split_imgs(self.split_type):
            labels_list.append(labels[target_img])
        return np.array(labels_list)

    def load_identities(self):
        identities = {}
        with open(self.identity_file_path, "r") as identity_file:
            info = identity_file.readlines()
            for line in info:
                img_id, _id = line.strip().split()
                identities[img_id] = int(_id)
        identities_list = []
        for target_img in self.get_target_split_imgs(self.split_type):
            identities_list.append(identities[target_img])
        return np.array(identities_list)

    def load_image_names(self):
        imgs = []
        for target_img in self.get_target_split_imgs(self.split_type):
            imgs.append(target_img)
        return np.array(imgs)

    def __getitem__(self, index):
        img_file_path = os.path.join(self.image_folder_path, self.img_names[index])
        label = self.labels[index]
        identity = self.identities[index]
        with open(img_file_path, "rb") as f:
            img = Image.open(f)
            img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return len(self.labels)


def load_celeba2(
    downsample_pct: float = 0.5, train_pct: float = 0.8, batch_size: int = 50, img_size: int = 32, label_list: list = None
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose([
                               transforms.Resize(img_size),
                               transforms.CenterCrop(img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_set = CelebADS(transform=transform, attr_list=label_list, split='train', downsample_pct=downsample_pct*train_pct)
    val_set = CelebADS(transform=transform, attr_list=label_list, split='val', downsample_pct=downsample_pct)
    test_set = CelebADS(transform=transform, attr_list=label_list, split='test', downsample_pct=downsample_pct)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    import csv
    CelebA_Image_root = os.path.join(CELEBA_ROOT, 'Img')
    CelebA_Attr_file = os.path.join(CELEBA_ROOT, 'Anno/list_attr_celeba.txt')
    with open(CelebA_Attr_file) as csv_file:
        data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))
    print(data[:3])
