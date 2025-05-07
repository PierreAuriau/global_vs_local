# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reduce the Champollion embedding dimensions with a ACP.
We reduced the dimensions from 256 to 20.
"""
import os
import pandas as pd
import numpy as np
import glob
from multiprocessing import Pool
import itertools
from sklearn.decomposition import PCA
from dataset import ClinicalDataset
from config import Config

config = Config()

def reduced(area, dataset):

    print(f"area: {area} | dataset: {dataset}")

    train_dataset = ClinicalDataset(dataset, area,
                                    split='train', 
                                    label=None, 
                                    transforms=None,
                                    target_mapping=None)
    mask = train_dataset._extract_mask(train_dataset.metadata,
                                       train_dataset._unique_keys)
    train_data = np.concatenate(train_dataset.images, axis=0)
    
    pca = PCA(n_components=config.n_components)

    pca.fit(train_data)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum()}")

    for images, study in zip(train_dataset.images,
                             train_dataset._studies):
        reduced = pca.transform(images)
        np.save(os.path.join(config.path2data, area,   
                        f"{study}_reduced_{area.lower()}.npy"),
        reduced.astype(np.float32))    


if __name__ == "__main__":
    serial = False
    parallel = True
    # Serial
    if serial:
        for area in config.areas:
            for dataset in config.datasets:
                reduced(area=area, dataset=dataset)
            
    # Parallel
    if parallel:
        with Pool() as pool:
            pool.starmap(reduced, 
                         [(a, s) for a,s in itertools.product(config.areas, 
                                                              config.datasets)])
        