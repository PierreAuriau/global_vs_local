# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import json
import re

import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, \
                            mean_squared_error, r2_score, roc_auc_score

# from project
from log import setup_logging
from model import BTModel
from datamanager import DataManager
from config import Config

config = Config()


def fit_bt_model(chkpt_dir,
                 n_embedding=256, nb_epochs=300, lr=1e-4,
                 correlation_bt="cross", lambda_bt=1.0,
                 data_augmentation="cutout", batch_size=32,
                 num_workers=8):
    
    model = BTModel(n_embedding=n_embedding, projector=True)
    datamanager = DataManager(dataset="ukb", label=None, two_views=True,  
                              data_augmentation=data_augmentation)
    
    train_loader = datamanager.get_dataloader(split="train", shuffle=True,
                                              batch_size=batch_size,
                                              num_workers=num_workers)
    val_loader = datamanager.get_dataloader(split="validation",
                                            batch_size=batch_size,
                                            num_workers=num_workers)
    
    model.fit(train_loader=train_loader, val_loader=None, 
              nb_epochs=nb_epochs,
              correlation_bt=correlation_bt, lambda_bt=lambda_bt,
              chkpt_dir=chkpt_dir,
              lr=lr)
    
def fine_tune_bt_model(chkpt_dir, dataset,
                       n_embedding=256, pretrained_epoch=299,
                       nb_epochs=100, lr=1e-4, weight_decay=5e-3,
                       batch_size=32, num_workers=8):
    
    model = BTModel(n_embedding=n_embedding, classifier=True)
    
    datamanager = DataManager(dataset=dataset, label="diagnosis", two_views=False,  
                              data_augmentation=None)  
    
    train_loader = datamanager.get_dataloader(split="train", shuffle=True,
                                            batch_size=batch_size,
                                            num_workers=num_workers)
    val_loader = datamanager.get_dataloader(split="validation",
                                            batch_size=batch_size,
                                            num_workers=num_workers)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(datamanager.dataset["train"].target.mean(), 
                                                           dtype=torch.float32,
                                                           device=model.device))

    model.fine_tuning(train_loader, val_loader, pretrained_epoch=pretrained_epoch, 
                        loss_fn=loss_fn, nb_epochs=nb_epochs, chkpt_dir=chkpt_dir, 
                        lr=lr, weight_decay=weight_decay,
                        logs={"dataset": dataset, "label": "diagnosis"})
    
    train_loader = datamanager.get_dataloader(split="train", shuffle=False,
                                    batch_size=config.batch_size,
                                    num_workers=config.num_workers)
    test_intra_loader = datamanager.get_dataloader(split="test_intra",
                                                    batch_size=batch_size,
                                                    num_workers=num_workers)
    test_loader = datamanager.get_dataloader(split="test",
                                             batch_size=batch_size,
                                             num_workers=num_workers)
    metrics = {
        "roc_auc": lambda y_true, y_pred: roc_auc_score(y_true=y_true, y_score=y_pred),
        "balanced_accuracy": lambda y_true, y_pred: balanced_accuracy_score( y_true=y_true,
                                                                            y_pred=y_pred.argmax(axis=1))
    }
    model.test_classifier(loaders=[train_loader, val_loader,
                                    test_intra_loader, test_loader],
                            splits=["train", "validation", "test_intra", "test"],
                            epoch=(nb_epochs-1), metrics=metrics, chkpt_dir=chkpt_dir,
                            logs={"dataset": dataset, "label": "diagnosis"})

def train(params):

    fit_bt_model(chkpt_dir=params["chkpt_dir"],
                 n_embedding=params.get("n_embedding", config.n_embedding),
                 nb_epochs=params.get("nb_epochs", config.nb_epochs),
                 lr=params.get("lr", config.lr),
                 correlation_bt=params.get("correlation", config.correlation_bt),
                 lambda_bt=params.get("lambda", config.lambda_bt),
                 data_augmentation=params.get("data_augmentation", config.data_augmentation),
                 batch_size=params.get("batch_size", config.batch_size),
                 num_workers=params.get("num_workers", config.num_workers))

def fine_tune(params):
    chkpt_dir = params["chkpt_dir"]
    with open(os.path.join(chkpt_dir, "hyperparameters.json"), "r") as json_file:
        hyperparameters = json.load(json_file)
    n_embedding = hyperparameters.get("n_embedding", config.n_embedding)
    epoch_f = hyperparameters.get("nb_epochs", config.nb_epochs) - 1
    
    for dataset in ("asd", "bd", "scz"):

        chkpt_dir_dt = os.path.join(chkpt_dir, dataset)
        os.makedirs(chkpt_dir_dt, exist_ok=True)
        if not os.path.exists(os.path.join(chkpt_dir_dt, f"barlowtwins_ep-{epoch_f}.pth")):
            os.symlink(os.path.join(chkpt_dir,
                                    f"barlowtwins_ep-{epoch_f}.pth"),
                    os.path.join(chkpt_dir_dt, f"barlowtwins_ep-{epoch_f}.pth"))
        
        fine_tune_bt_model(chkpt_dir_dt, dataset,
                           n_embedding=n_embedding, 
                           pretrained_epoch=epoch_f,
                           nb_epochs=params.get("nb_epochs", config.nb_epochs_ft), 
                           lr=params.get("lr", config.lr_ft), 
                           weight_decay=params.get("weight_decay", config.weight_decay_ft),
                           batch_size=params.get("batch_size", config.batch_size), 
                           num_workers=params.get("num_workers", config.num_workers))

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--chkpt_dir", required=True, type=str,
    help="Checkpoint dir where all the logs are stored. List of existing checkpoint directories: " \
        + " - ".join(os.listdir(config.path2models)))
    parser.add_argument("-t", "--train", action="store_true", help="Train the model")
    parser.add_argument("-f", "--fine_tune", action="store_true", help="Finetune the model")
    args, unknownargs = parser.parse_known_args(argv)
    params = {}
    for i in range(0, len(unknownargs), 2):
        key = re.search("--([a-z_]+)", unknownargs[i])[1]
        params[key] = eval(unknownargs[i+1])
    return args, params


if __name__ == "__main__":

    args, params = parse_args(sys.argv[1:])
    chkpt_dir = os.path.join(config.path2models, args.chkpt_dir)
    params["chkpt_dir"] = chkpt_dir
    os.makedirs(chkpt_dir, exist_ok=True)
    setup_logging(level="info", 
                  logfile=os.path.join(chkpt_dir, "train_bt_model.log"))
    
    if args.train:
        train(params)
    if args.fine_tune:
        fine_tune(params)
    