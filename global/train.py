# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse

# from project
from log import setup_logging
from model import BTModel
from datamanager import DataManager
from config import Config

config = Config()


def train_bt_model(chkpt_dir, lambda_bt, correlation_bt):
    
    model = BTModel(n_embedding=256, projector=True)
    datamanager = DataManager(dataset="ukb", label=None, two_views=True,  
                              batch_size=32, data_augmentation="cutout",
                              num_workers=8, pin_memory=True)
    
    train_loader = datamanager.get_dataloader(split="train")
    val_loader = datamanager.get_dataloader(split="validation")
    
    model.fit(train_loader=train_loader, val_loader=val_loader, 
              nb_epochs=100,
              correlation=correlation_bt, lambda_param=lambda_bt,
              chkpt_dir=chkpt_dir)
    
def test_bt_model(chkpt_dir):
    model = BTModel(n_embedding=256, projector=False)
    datamanager = DataManager(dataset="ukb", label="Sex", two_views=False,  
                              batch_size=32, data_augmentation=None,
                              num_workers=8)
    
    train_loader = datamanager.get_dataloader(split="train")
    val_loader = datamanager.get_dataloader(split="validation")
    
    model.test_linear_probe(train_loader=train_loader, val_loader=val_loader,
                            label="sex",
                            epochs=[i for i in range(0, 100, 10)] + [99], 
                            chkpt_dir=chkpt_dir)
    
def test_bt_model_dx(chkpt_dir):

    model = BTModel(n_embedding=256, projector=False)

    for dataset in ("asd", "bd", "scz"):
        datamanager = DataManager(dataset=dataset, label="diagnosis", two_views=False,  
                                batch_size=32, data_augmentation=None,
                                num_workers=8)
        chkpt_dir_dt = os.path.join(chkpt_dir, dataset)
        os.makedirs(chkpt_dir_dt, exist_ok=True)
        list_epochs = [i for i in range(0, 100, 10)] + [99]
        for epoch in list_epochs:
            if not os.path.exists(os.path.join(chkpt_dir_dt, 
                                               f"barlowtwins_ep-{epoch}.pth")):
                os.symlink(os.path.join(chkpt_dir,
                                    f"barlowtwins_ep-{epoch}.pth"),
                        os.path.join(chkpt_dir_dt, f"barlowtwins_ep-{epoch}.pth"))
        train_loader = datamanager.get_dataloader(split="train")
        val_loader = datamanager.get_dataloader(split="validation")
    
        model.test_linear_probe(train_loader=train_loader, val_loader=val_loader,
                                label="diagnosis",
                                epochs=list_epochs, 
                                chkpt_dir=chkpt_dir_dt)
    
def fine_tune_model(chkpt_dir):
    
    model = BTModel(n_embedding=256, classifier=True)

    for dataset in ("asd", "bd"):
        datamanager = DataManager(dataset="scz", label="diagnosis", two_views=False,  
                                batch_size=32, data_augmentation=None,
                                num_workers=8)
        chkpt_dir_dt = os.path.join(chkpt_dir, dataset)
        os.makedirs(chkpt_dir_dt, exist_ok=True)
        if not os.path.exists(os.path.join(chkpt_dir_dt, "barlowtwins_ep-99.pth")):
            os.symlink(os.path.join(config.path2models, "20241017_correct_lambda_param",
                                    "barlowtwins_ep-99.pth"),
                    os.path.join(chkpt_dir_dt, "barlowtwins_ep-99.pth"))  
        train_loader = datamanager.get_dataloader(split="train")
        val_loader = datamanager.get_dataloader(split="validation")
        test_intra_loader = datamanager.get_dataloader(split="test_intra")
        test_loader = datamanager.get_dataloader(split="test")
        
        model.fine_tuning(train_loader, val_loader, pretrained_epoch=99, 
                          nb_epochs=100, chkpt_dir=chkpt_dir_dt, 
                          lr=1e-4, weight_decay=5e-5)
        
        model.test_classifier(loaders=[train_loader, val_loader,
                                       test_intra_loader, test_loader],
                             splits=["train", "validation", "test_intra", "test"],
                             epoch=99, chkpt_dir=chkpt_dir_dt)
    
def test_classifier(chkpt_dir):
    model = BTModel(n_embedding=256, classifier=True)
    datamanager = DataManager(dataset="scz", label="diagnosis", two_views=False,  
                              batch_size=32, data_augmentation=None,
                              num_workers=8)
    splits = ["train", "validation", "test_intra", "test"]
    loaders = [datamanager.get_dataloader(split=s) for s in splits]
    
    model.test_classifier(loaders=loaders, splits=splits, 
                          epoch=99, chkpt_dir=chkpt_dir)
    
def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--chkpt_dir", required=True, type=str,
    help="Checkpoint dir where all the logs are stored. List of existing checkpoint directories:" \
        + "/".join(os.listdir(config.path2models)))
    parser.add_argument("-e", "--exp", required=True, type=str,
    help="Experience that you want to launch")
    args = parser.parse_args(argv)
    return args


if __name__ == "__main__":

    args = parse_args(sys.argv[1:])
    chkpt_dir = os.path.join(config.path2models, args.chkpt_dir)
    os.makedirs(chkpt_dir, exist_ok=True)
    setup_logging(level="info", 
                  logfile=os.path.join(chkpt_dir, "train_bt_model.log"))
    if args.exp == "fine_tune_model":
        fine_tune_model(chkpt_dir=chkpt_dir)
    elif args.exp == "test_bt_model":
        test_bt_model(chkpt_dir=chkpt_dir)
    elif args.exp == "test_classifier":
        test_classifier(chkpt_dir=chkpt_dir)
    elif args.exp == "test_bt_model_dx":
        test_bt_model_dx(chkpt_dir=chkpt_dir)
    