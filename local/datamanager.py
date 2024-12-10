# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Import
import numpy as np
import pandas as pd
import logging
from typing import List, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# project imports
from dataset import ClinicalDataset


class ToTensor(nn.Module):
    def forward(self, arr):
        arr = np.expand_dims(arr, axis=0)
        return torch.from_numpy(arr)


class ToArray(nn.Module):
    def forward(self, tensor):
        tensor = tensor.squeeze()
        return np.asarray(tensor)


class DataManager(object):

    def __init__(self, dataset: str, area: str,
                 label: str = None, reduced: bool = False):
        
        self.logger = logging.getLogger("datamanager")
        self.label = label

        if label == "sex":
            target_mapping = {"H": 0, "F": 1}
        elif label == "diagnosis":
            target_mapping = {"control": 0,     
                              "asd": 1,
                              "bd": 1, "bipolar disorder": 1, "psychotic bd": 1, 
                              "scz": 1}
        else:
            target_mapping = None
        
        tr = ToTensor()
        
        self.dataset = dict()
        self.dataset["train"] = ClinicalDataset(split="train", label=label, area=area,
                                                dataset=dataset, transforms=tr,
                                                target_mapping=target_mapping,
                                                reduced=reduced)
        self.dataset["validation"] = ClinicalDataset(split="validation", label=label, area=area,
                                                     dataset=dataset, transforms=tr,
                                                     target_mapping=target_mapping,
                                                     reduced=reduced)
        self.dataset["test_intra"] = ClinicalDataset(split="test_intra", label=label, area=area,
                                                     dataset=dataset, transforms=tr,
                                                     target_mapping=target_mapping,
                                                     reduced=reduced)
        self.dataset["test"] = ClinicalDataset(split="test", label=label, area=area,
                                               dataset=dataset, transforms=tr,
                                               target_mapping=target_mapping,
                                               reduced=reduced)

    def get_dataloader(self, split, batch_size, **kwargs):
        dataset = self.dataset[split]
        drop_last = True if len(dataset) % batch_size == 1 else False
        if drop_last:
            self.logger.warning(f"The last subject of the {split} set will not be feed into the model ! "
                                f"Change the batch size ({batch_size}) to keep all subjects ({len(dataset)})")
        loader = DataLoader(dataset, batch_size=batch_size,
                            drop_last=drop_last, **kwargs)
        loader.split = split # FIXME : to keep ?
        return loader
        
    def __str__(self):
        return "DataManager"
    
if __name__ == "__main__":
    # test
    datamanager = DataManager(dataset="bd", area="SC-SPoC_left",
                              label="diagnosis", batch_size=32)
    for split in ("train", "validation", "test_intra", "test"):
        print("# Split:", split)
        print("nb sbj:", len(datamanager.dataset[split]))
        loader = datamanager.get_dataloader(split=split)
        print("nb batch:", len(loader))
        for i, sample in enumerate(loader):
            print(split, i)
            pass
            """
            print(sample["input"].size())
            print(sample["label"].size())
            print(sample["input"][0])
            break
            """