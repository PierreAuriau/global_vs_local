# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import pandas as pd
import numpy as np

from typing import Callable, List, Type, Sequence, Dict, Union
import torch
from torch.utils.data.dataset import Dataset

from config import Config

logger = logging.getLogger("dataset")
config = Config()

class UKBDataset(Dataset):

    def __init__(self, split: str = 'train', 
                 label: str = None, 
                 transforms: Callable[[np.ndarray], np.ndarray] = None, 
                 two_views: bool = False):
        """
        :param target: str or [str], either 'sex' or 'age'.
        :param split: str, either 'train', 'validation'
        :param transforms: Callable, data transformations
        :param two_views: bool, return two views of each item
        """
        # 0) set attributes
        self.split = split
        self.transforms = transforms
        self.two_views = two_views
                
        # 1) Loads globally all the data
        self.metadata = pd.read_csv(os.path.join(config.path2data, "ukbiobank_t1mri_skeleton_participants.csv"), dtype=self._id_types)
        self.scheme = self.load_scheme()

        # 2) Selects the data to load in memory according to selected scheme
        mask = self._extract_mask(self.metadata, unique_keys=self._unique_keys, check_uniqueness=True)
        self.metadata = self.metadata[mask]

        # 3) Get the labels to predict
        if label is not None:
            self.label = label
            assert self.label in self.metadata.colums(), \
                f"Inconsistent files: missing {self.label} in participants DataFrame"
            self.target = self.metadata[self.label].values.astype(np.float32)
            assert self.target.isna().sum().sum() == 0, f"Missing values in {self.label} column"
        else:
            self.target = None

    @property
    def _train_val_scheme(self) -> str:
        return "ukbiobank_train_validation_subjects.csv"

    @property
    def _unique_keys(self) -> List[str]:
        return ["participant_id"]
    
    @property
    def _id_types(self):
        return {"participant_id": str,
                "session": int,
                "acq": int,
                "run": int}

    def _extract_mask(self, df: pd.DataFrame, unique_keys: Sequence[str], check_uniqueness: bool = True):
        """
        :param df: pandas DataFrame
        :param unique_keys: list of str
        :param check_uniqueness: if True, check the unique_keys identified uniquely an image in the dataset
        :return: a binary mask indicating, for each row, if the participant belongs to the current scheme or not.
        """
        _source_keys = df[unique_keys].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        if check_uniqueness:
            assert len(set(_source_keys)) == len(_source_keys), f"Multiple identique identifiers found"
        _target_keys = self.scheme[unique_keys].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        mask = _source_keys.isin(_target_keys).values.astype(bool)
        return mask
    
    def load_scheme(self):
        scheme_df = pd.read_csv(os.path.join(config.path2schemes, self._train_val_scheme), dtype=self._id_types)
        return scheme_df.loc[scheme_df["set"] == self.split, self._unique_keys]
    
    def __getitem__(self, idx: int):
        sample = dict()
        if self.target is not None:
            sample[self.label] = self.target[idx]

        arr_path = self.metadata["arr_path"].iloc[idx]
        arr = np.load(arr_path)
        if self.two_views:
            sample["view_1"] = self.transforms(arr.copy())
            sample["view_2"] = self.transforms(arr.copy())
        else:
            if self.transforms is not None:
                sample["input"] = self.transforms(arr)
            else:
                sample["input"] = arr
        return sample
    
    def __len__(self):
        return len(self.metadata)

    def __str__(self):
        return f"UKBDataset({self.split} set)"
    
if __name__ == "__main__":
    train_dataset = UKBDataset(split="train")
    val_dataset = UKBDataset(split="validation")
    assert (len(train_dataset) + len(val_dataset) == 21045), "Wrong number of subjects"
    item = train_dataset[124]
    print(item.keys())
    assert np.all(item["input"].shape == (128, 160, 128)), "Wrong image shape"
    assert set(np.unique(item["input"])).issubset({0, 1}), "Wrong values in skeleton"
    assert item["input"].dtype == np.float32, "Wrong data type"
