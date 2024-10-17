# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Import
import numpy as np
import pandas as pd
import logging
from typing import List, Union

import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

# project imports
from dataset import UKBDataset
from data_augmentation import Cutout, Shift, Blur, ToTensor


class DataManager(object):

    def __init__(self, label: str = None, two_views: bool = False,  
                 batch_size: int = 1, data_augmentation: str = None,
                 **dataloader_kwargs):
        
        self.logger = logging.getLogger("datamanager")
        self.two_views = two_views
        self.batch_size = batch_size
        self.label = label
        self.dataloader_kwargs = dataloader_kwargs

        if data_augmentation == "cutout":
            tr = transforms.Compose([Cutout(patch_size=0.4, random_size=True,
                                                   localization="on_data", min_size=0.1),
                                            ToTensor()])
        elif data_augmentation == "shift":
            tr = transforms.Compose([Shift(nb_voxels=1, random=True),
                                            Cutout(patch_size=0.4, random_size=True,
                                                   localization="on_data", min_size=0.1),
                                            ToTensor()])
        elif data_augmentation == "blur":
            tr = transforms.Compose([Blur(sigma=1.0),
                                            Cutout(patch_size=0.4, random_size=True,
                                                   localization="on_data", min_size=0.1),
                                            ToTensor()])
        elif data_augmentation == "all":
            tr = transforms.Compose([transforms.RandomApply([Blur(sigma=1.0),
                                                             Shift(nb_voxels=1, random=True)],
                                                            p=0.5),
                                     Cutout(patch_size=0.4, random_size=True,
                                            localization="on_data", min_size=0.1),
                                     ToTensor()])
        else:
            tr = ToTensor()

        self.dataset = dict()
        self.dataset["train"] = UKBDataset(split='train', label=label, 
                                           transforms=tr, two_views=two_views)
        self.dataset["validation"] = UKBDataset(split='validation', label=label, 
                                                transforms=tr, two_views=two_views)
    
    def get_dataloader(self, split):
        dataset = self.dataset[split]
        drop_last = True if len(dataset) % self.batch_size == 1 else False
        if drop_last:
            self.logger.warning(f"The last subject of the {split} set will not be feed into the model ! "
                                f"Change the batch size ({self.batch_size}) to keep all subjects ({len(dataset)})")
        if split == "train":
            shuffle = True
        else:
            shuffle = False
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle,
                            drop_last=drop_last, **self.dataloader_kwargs)
        return loader
        
    def __str__(self):
        return "DataManager"
    
if __name__ == "__main__":
    datamanager = DataManager(label=None, two_views=False, batch_size=32, data_augmentation=None)
    train_loader = datamanager.get_dataloader(split="train")
    val_loader = datamanager.get_dataloader(split="validation")
    for sample in val_loader:
        break
    assert len(train_loader) == 592
    assert (sample["input"].size() == torch.Tensor(32, 1, 128, 160, 128))
    from data_augmentation import ToArray
    np.save("/neurospin/dico/pauriau/tmp/view_1.npy", ToArray()(sample["view_1"]))
    np.save("/neurospin/dico/pauriau/tmp/view_2.npy", ToArray()(sample["view_2"]))
