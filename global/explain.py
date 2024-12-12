# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re
import json
import argparse
import itertools
import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, \
                            mean_squared_error, r2_score, roc_auc_score
import matplotlib.pyplot as plt

from torchvision.transforms import transforms
from torch.utils.data import DataLoader

# project imports
from dataset import ClinicalDataset
from data_augmentation import ToTensor, Occlusion, Cutout

# from project
from log import setup_logging
from model import BTModel
from config import Config

config = Config()

def explain_classifier(chkpt_dir, epoch=-1):
    
    with open(os.path.join(chkpt_dir, "hyperparameters.json"), "r") as json_file:
        hyperparameters = json.load(json_file)
    n_embedding = hyperparameters.get("n_embedding", config.n_embedding)
    model = BTModel(n_embedding=n_embedding, classifier=True)
    
    metrics = {
        "roc_auc": lambda y_true, y_pred: roc_auc_score(y_true=y_true, y_score=y_pred),
        "balanced_accuracy": lambda y_true, y_pred: balanced_accuracy_score( y_true=y_true,
                                                                                y_pred=y_pred.argmax(axis=1))}
    fold = None
    label = "diagnosis"
    splits = ["internal_test", "external_test"]
    target_mapping = {"control": 0, 
                    "asd": 1,
                    "bd": 1, "bipolar disorder": 1, "psychotic bd": 1, 
                    "scz": 1}


    for dataset in ("asd", "bd", "scz"):
        chkpt_dir_dt = os.path.join(chkpt_dir, dataset)
        with open(os.path.join(chkpt_dir_dt, "hyperparameters.json"), "r") as json_file:
            hyperparameters = json.load(json_file)
        if epoch < 0:
            epoch = hyperparameters.get("nb_epochs", config.nb_epochs_ft) + epoch

        for area in config.areas:

            chkpt_dir_ae = os.path.join(chkpt_dir, dataset, area)
            os.makedirs(chkpt_dir_ae, exist_ok=True)
            if not os.path.exists(os.path.join(chkpt_dir_ae, f"classifier_ep-{epoch}.pth")):
                os.symlink(os.path.join(chkpt_dir_dt, f"classifier_ep-{epoch}.pth"),
                           os.path.join(chkpt_dir_ae, f"classifier_ep-{epoch}.pth"))

            tr = transforms.Compose([Occlusion(area), ToTensor()])

            datasets = [ClinicalDataset(split=s, label=label, fold=fold,
                                        dataset=dataset, transforms=tr,
                                         target_mapping=target_mapping)
                        for s in splits]
            
            loaders = [DataLoader(d, drop_last=False, batch_size=config.batch_size, 
                                  num_workers=config.num_workers)
                        for d in datasets]

        
            model.test_classifier(loaders=loaders, splits=splits, 
                                  epoch=epoch, metrics=metrics, chkpt_dir=chkpt_dir_ae,
                                  logs={"dataset": dataset, "label": "diagnosis", "occluded_area": area})


def patch_occlusion(chkpt_dir, epoch=-1):
    
    with open(os.path.join(chkpt_dir, "hyperparameters.json"), "r") as json_file:
        hyperparameters = json.load(json_file)
    n_embedding = hyperparameters.get("n_embedding", config.n_embedding)
    model = BTModel(n_embedding=n_embedding, classifier=True)
    
    metrics = {
        "roc_auc": lambda y_true, y_pred: roc_auc_score(y_true=y_true, y_score=y_pred),
        "balanced_accuracy": lambda y_true, y_pred: balanced_accuracy_score(y_true=y_true,
                                                                            y_pred=(y_pred > 0.5).astype(int))
    }
    patch_size = 16
    step_size = 4
    img_size = (128, 160, 128)
    fold = None
    label = "diagnosis"
    splits = ["internal_test", "external_test"]
    target_mapping = {"control": 0, 
                    "asd": 1,
                    "bd": 1, "bipolar disorder": 1, "psychotic bd": 1, 
                    "scz": 1}
    
    for dataset in ("asd", "bd", "scz"):
        print(f"DATASET: {dataset}")
        chkpt_dir_dt = os.path.join(chkpt_dir, dataset)
        with open(os.path.join(chkpt_dir_dt, "hyperparameters.json"), "r") as json_file:
            hyperparameters = json.load(json_file)
        if epoch < 0:
            epoch = hyperparameters.get("nb_epochs", config.nb_epochs_ft) + epoch
        
        for x, y, z in itertools.product(range(patch_size//2, img_size[0]-patch_size//2, step_size), 
                                    range(patch_size//2, img_size[1]-patch_size//2, step_size), 
                                    range(patch_size//2, img_size[2]-patch_size//2, step_size)):
            if dataset == "asd":
                if x < 96:
                    continue
            
            print(f"Patch ({x}, {y}, {z})")
            chkpt_dir_xyz = os.path.join(chkpt_dir, dataset, "xai", f"x-{x}_y-{y}_z-{z}")
            os.makedirs(chkpt_dir_xyz, exist_ok=True)
            if not os.path.exists(os.path.join(chkpt_dir_xyz, f"classifier_ep-{epoch}.pth")):
                os.symlink(os.path.join(chkpt_dir_dt, f"classifier_ep-{epoch}.pth"),
                           os.path.join(chkpt_dir_xyz, f"classifier_ep-{epoch}.pth"))

            tr = transforms.Compose([Cutout(patch_size=patch_size, localization=(x, y, z), random_size=False), ToTensor()])

            datasets = [ClinicalDataset(split=s, label=label, fold=fold,
                                        dataset=dataset, transforms=tr,
                                        target_mapping=target_mapping)
                        for s in splits]
            
            loaders = [DataLoader(d, drop_last=False, batch_size=30, 
                                  num_workers=config.num_workers)
                        for d in datasets]

        
            model.test_classifier(loaders=loaders, splits=splits, 
                                  epoch=epoch, metrics=metrics, chkpt_dir=chkpt_dir_xyz,
                                  logs={"dataset": dataset, "label": "diagnosis", 
                                        "x_occluded": x, "y_occluded": y, "z_occluded": z})

if __name__ == "__main__":
    """ Test dataset cutout
    dataset = "bd"
    patch_size = 16
    step_size = 4
    img_size = (128, 160, 128)
    fold = None
    label = "diagnosis"
    split = "internal_test"
    target_mapping = {"control": 0, 
                    "asd": 1,
                    "bd": 1, "bipolar disorder": 1, "psychotic bd": 1, 
                    "scz": 1}

    
    import itertools
    for x, y, z in itertools.product(range(patch_size//2, img_size[0]-patch_size//2, step_size), 
                                     range(patch_size//2, img_size[1]-patch_size//2, step_size), 
                                     range(patch_size//2, img_size[2]-patch_size//2, step_size)):
        if (x < 60) or (y < 60) or (z < 60):
            continue

        tr = transforms.Compose([Cutout(patch_size=patch_size, localization=(x, y, z), random_size=False)])

        clinicaldataset = ClinicalDataset(dataset, split=split, label=label, fold=fold,
                                  transforms=tr, target_mapping=target_mapping)

        
        img = clinicaldataset[12]["input"]

        if (x > 80):
            break

        np.save(f"/neurospin/dico/pauriau/tmp/xai/img-12_x-{x}_y-{y}_z-{z}.npy", img)
    """
    patch_occlusion(chkpt_dir="/neurospin/psy_sbox/analyses/2024_pauriau_global_vs_local/models/global/20241106_without_ventricle_bt_model_256")