# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
import numpy as np

# from project
from log import setup_logging
from model import DLModel
from datamanager import DataManager
from config import Config

config = Config()


logger = logging.getLogger("train")
   
def train_dl_model(chkpt_dir, dataset, area):
    
    chkpt_dir = os.path.join(chkpt_dir, dataset, area)
    os.makedirs(chkpt_dir, exist_ok=True)
    setup_logging(level="info", 
                  logfile=os.path.join(chkpt_dir, "train_dl_model.log"))
    
    model = DLModel()

    datamanager = DataManager(dataset=dataset, area=area,
                              label="diagnosis")
    train_loader = datamanager.get_dataloader(split="train",
                                              batch_size=config.batch_size,
                                              shuffle=True, 
                                              num_workers=config.num_workers)
    val_loader = datamanager.get_dataloader(split="validation",
                                            batch_size=60,
                                            num_workers=config.num_workers)
    test_intra_loader = datamanager.get_dataloader(split="test_intra",
                                                   batch_size=60,
                                                   num_workers=config.num_workers)
    test_loader = datamanager.get_dataloader(split="test",
                                             batch_size=60,
                                             num_workers=config.num_workers)
        
    model.fit(train_loader, val_loader,
              nb_epochs=config.nb_epochs, chkpt_dir=chkpt_dir,
              logs={"area": area, "dataset": dataset},
              lr=config.lr, weight_decay=config.weight_decay)
    
    model.test(loaders=[train_loader, val_loader,
                        test_intra_loader, test_loader],
                splits=["train", "validation", "test_intra", "test"],
                epoch=config.epoch_f, chkpt_dir=chkpt_dir, save_y_pred=True,
                logs={"area": area, "dataset": dataset})

def train_all_dl_models(chkpt_dir):
    chkpt_dir = os.path.join(config.path2models, chkpt_dir)
    for dataset in config.datasets:
        logger.info(f"\n# DATASET: {dataset}\n" + "-"*(len(dataset)+11))
        for area in config.areas:
            logger.info(f"\n## AREA: {area}\n"+ "-"*(len(area)+10))
            train_dl_model(chkpt_dir=chkpt_dir,
                           dataset=dataset,
                           area=area)


def test_l2_regularisation():
    chkpt_dir = os.path.join(config.path2models, "20241101_test_l2_regularisation")
    os.makedirs(chkpt_dir, exist_ok=True)
    setup_logging(level="info", logfile=os.path.join(chkpt_dir, 
                                                     "train_dl_model.log"))
    for dataset in ("scz", "bd"):
        for area in config.areas[10:15]:
            datamanager = DataManager(dataset=dataset, area=area,
                                      label="diagnosis")
            chkpt_dir_dt = os.path.join(chkpt_dir, dataset, area)
            os.makedirs(chkpt_dir_dt, exist_ok=True)
            train_loader = datamanager.get_dataloader(split="train",
                                                      shuffle=True,
                                                      batch_size=64, 
                                                      num_workers=8)
            val_loader = datamanager.get_dataloader(split="validation",
                                                    batch_size=60, 
                                                    num_workers=8)
            test_intra_loader = datamanager.get_dataloader(split="test_intra",
                                                           batch_size=60, 
                                                           num_workers=8)
            test_loader = datamanager.get_dataloader(split="test",
                                                     batch_size=60, 
                                                     num_workers=8)
            for weight_decay in (5e-4, 5e-3, 5e-2, 5e-1, 5, 5e1, 5e2):
                print("\n" + "-"*(35+len(dataset)+len(area)+len(str(weight_decay))) +"\n")
                print(f"dataset: {dataset} - area: {area} - weight_decay: {weight_decay}")
                print("\n" + "-"*(35+len(dataset)+len(area)+len(str(weight_decay))) +"\n")
                chkpt_dir_wd = os.path.join(chkpt_dir, dataset, area, f"wd-{weight_decay}")
                os.makedirs(chkpt_dir_wd, exist_ok=True)
                model = DLModel()
                model.fit(train_loader, val_loader,
                        nb_epochs=100, chkpt_dir=chkpt_dir_wd,
                        logs={"area": area, 
                              "weight_decay": weight_decay,
                              "dataset": dataset},
                        lr=1e-4, weight_decay=weight_decay)
                
                model.test(loaders=[train_loader, val_loader,
                                    test_intra_loader, test_loader],
                            splits=["train", "validation", "test_intra", "test"],
                            epoch=99, chkpt_dir=chkpt_dir_wd, save_y_pred=False,
                            logs={"area": area,
                                  "weight_decay": weight_decay,
                                  "dataset": dataset})

def test_linear_probe():
    
    for dataset in config.datasets:
        epoch = 99
        chkpt_dir = os.path.join(config.path2models, 
                                "20241102_local_models_high_weight_decay", 
                                dataset)
        model = DLModel()
        datamanager = DataManager(dataset=dataset, area=config.areas[0],
                                  label="diagnosis")
        predictions = {}
        labels = {}
        for split in ("train", "validation", "test_intra", "test"):
            predictions[split] = np.stack([
                np.load(os.path.join(chkpt_dir,
                                    area,
                                    f"y_pred_ep-{epoch}_set-{split}.npy"))
                for area in config.areas], 
                axis=1)
            # in case of drop_last
            labels[split] = datamanager.dataset[split].target[:len(predictions[split])]

        model.test_linear_probe(predictions=predictions,
                                labels=labels,
                                epoch=epoch,
                                chkpt_dir=chkpt_dir,
                                logs={"label": "diagnosis",
                                    "dataset": dataset},
                                save_y_pred=True)



def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--chkpt_dir", required=True, type=str,
    help="Checkpoint dir where all the logs are stored. List of existing checkpoint directories: " \
        + " - ".join(os.listdir(config.path2models)))
    # parser.add_argument("-d", "--dataset", required=True, type=str,
    # help="Dataset on which you want to train")
    args = parser.parse_args(argv)
    return args


if __name__ == "__main__":
    
    args = parse_args(sys.argv[1:])
    # train_all_dl_models(chkpt_dir=args.chkpt_dir)
    # test_l2_regularisation()
    test_linear_probe()