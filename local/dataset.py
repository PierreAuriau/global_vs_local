# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import pickle
import pandas as pd
import numpy as np
import bisect
from typing import Callable, List, Type, Sequence, Dict, Union
import torch
from torch.utils.data.dataset import Dataset

from config import Config

logger = logging.getLogger("dataset")
config = Config()


class ClinicalDataset(Dataset):

    def __init__(self, dataset: str, area: str, split: str = 'train', 
                 label: str = "diagnosis", fold: int = None,
                 transforms: Callable[[np.ndarray], np.ndarray] = None,
                 target_mapping: dict = None, reduced: bool = None,):
        """
        :param dataset: str, either 'asd', 'bd' or 'scz'
        :param area: strn one of the brain area
        :param label: str, either 'diagnosis', 'sex' or 'age'.
        :param split: str, either 'train', 'validation'
        :param transforms: Callable, data transformations
        :param target_mapping: dict, mapping between labels and int to predict
        :param reduced: bool, used reduced components by PCA or not
        """
        # 0) set attributes
        self.dataset = dataset
        self.area = area
        self.split = split
        self.fold = fold
        self.transforms = transforms
        self.reduced = reduced

        # 1) Loads globally all the data
        self.metadata = pd.concat([pd.read_csv(os.path.join(config.path2data, area,
                                                            f"{s}_skeleton_{area.lower()}_participants.csv"),
                                                            dtype=config.id_types) 
                                   for s in self._studies],
                       ignore_index=True, sort=False)
        self.scheme = self.load_scheme()
        filename = "reduced" if reduced else "skeleton"
        self.images = [np.load(os.path.join(config.path2data, area,
                                            f"{s}_{filename}_{area.lower()}.npy"), mmap_mode='r')
                       for s in self._studies]
        
        if len(self.metadata) != np.sum([len(arr) for arr in self.images]):
            import pdb
            pdb.set_trace()
        # 2) Selects the data to load in memory according to selected scheme
        mask = self._extract_mask(self.metadata, unique_keys=self._unique_keys, check_uniqueness=True)
        self.metadata = self.metadata[mask]

        # 3) Get the labels to predict
        if label is not None:
            self.label = label
            assert self.label in self.metadata.columns, \
                f"Inconsistent files: missing {self.label} in participants DataFrame"
            self.target = self.metadata[self.label]
            if target_mapping is not None:
                self.target = self.target.replace(target_mapping)
            assert self.target.isna().sum().sum() == 0, f"Missing values in {self.label} column"
            self.target = self.target.values.astype(np.float32)
        else:
            self.target = None

        # Attributes for mapping idx
        self._mask_indices = np.arange(len(mask))[mask]
        self._cumulative_sizes = np.cumsum([len(arr) for arr in self.images])

    @property
    def _studies(self) -> List:
        if self.dataset == "asd":
            return ["abide1", "abide2"]
        elif self.dataset == "bd":
            return ["biobd", "bsnip1", "cnp", "candi"]
        elif self.dataset == "scz":
            return ["schizconnect-vip-prague", "bsnip1", "cnp", "candi"]

    @property
    def _train_val_scheme(self) -> str:
        return f"{self.dataset}_age_sex_diagnosis_site_stratified_10-fold.csv"

    @property
    def _unique_keys(self) -> List[str]:
        if self.dataset == "asd":
            return ["participant_id", "session", "run", "study"]
        elif self.dataset in ["bd", "scz"]:
            return ["participant_id", "session", "study"]
    
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
        """ Old version
        with open(os.path.join(config.path2schemes, self._train_val_scheme), "rb") as f:
            scheme = pickle.load(f)
        return scheme[self.split]"""
        scheme = pd.read_csv(os.path.join(config.path2schemes, self._train_val_scheme), dtype=self._id_types)
        if self.fold is None:
            return scheme.loc[scheme["set"] == self.split, self._unique_keys]
        else:
            return scheme.loc[scheme[f"fold-{self.fold}"] == self.split, self._unique_keys]
    
    
    def _mapping_idx(self, idx: int):
        """
        :param idx: int ranging from 0 to len(dataset)-1
        :return: integer that corresponds to the original image index to load
        """
        idx = self._mask_indices[idx]
        dataset_idx = bisect.bisect_right(self._cumulative_sizes, idx)
        sample_idx = idx - self._cumulative_sizes[dataset_idx - 1] if dataset_idx > 0 else idx
        return (dataset_idx, sample_idx)
    
    def __getitem__(self, idx: int):
        sample = dict()
        if self.target is not None:
            sample["label"] = self.target[idx]
        (dataset_idx, sample_idx) = self._mapping_idx(idx)
        arr = self.images[dataset_idx][sample_idx]
        if self.transforms is not None:
            sample["input"] = self.transforms(arr.copy())
        else:
            sample["input"] = arr.copy()
        return sample
    
    def __len__(self):
        return len(self.metadata)

    def __str__(self):
        return f" {self.dataset.upper()}Dataset({self.split} set)"


if __name__ == "__main__":

    for area in config.areas:
    
        for split, n in zip(config.splits,
                            (1299, 163, 161, 116)):
            asddataset = ClinicalDataset(dataset="asd", split=split, area="SC-sylv_left",
                                        target_mapping={"asd": 1, "control": 0})
            # assert len(asddataset) == n, "Wrong number of subjects"
            item = asddataset[54]
            assert np.all(set(asddataset.target).issubset((0, 1))), "Wrong metadata name"
            assert np.all(item["input"].shape == (256, )), "Wrong image shape"
            assert item["input"].dtype == np.float32, "Wrong data type"
            assert not np.isnan(item["input"]).any(axis=(0, 1, 2)), "Found NaN in input"
        
        for split, n in zip(config.splits,
                            (831, 101, 106, 131)):
            bddataset = ClinicalDataset(dataset="bd", split=split, area="SC-SPoC_left",
                                        target_mapping={"bd": 1, "psychotic bd": 1, "bipolar disorder": 1, "control": 0})
            # assert len(bddataset) == n, "Wrong number of subjects"
            item = bddataset[54]
            assert np.all(set(bddataset.target).issubset((0, 1))), "Wrong metadata name"
            assert np.all(item["input"].shape == (256, )), "Wrong image shape"
            assert item["input"].dtype == np.float32, "Wrong data type"
        
        for split, n in zip(config.splits,
                            (928, 101, 118, 130)):
            sczdataset = ClinicalDataset(dataset="scz", split=split, area="SC-sylv_left",
                                        target_mapping={"scz": 1, "control": 0})
            # assert len(sczdataset) == n, "Wrong number of subjects"
            item = sczdataset[54]
            assert np.all(set(sczdataset.target).issubset((0, 1))), "Wrong metadata name"
            assert np.all(item["input"].shape == (256, )), "Wrong image shape"
            assert item["input"].dtype == np.float32, "Wrong data type"
