# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import json
import re
import logging

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
                       n_embedding=256, pretrained_epoch=299, refit=False,
                       fold=None, nb_epochs=100, lr=1e-4, weight_decay=5e-3,
                       batch_size=32, num_workers=8):
    
    logger.debug(f"Refit: {refit}")
    logger.debug(f"Learning rate: {lr}")
    logger.debug(f"Pretrained epoch: {pretrained_epoch}")
    logger.debug(f"Weight decay: {weight_decay}")
    
    model = BTModel(n_embedding=n_embedding, classifier=True)
    
    datamanager = DataManager(dataset=dataset, label="diagnosis", fold=fold,
                              two_views=False, data_augmentation=None)  
    
    train_loader = datamanager.get_dataloader(split="train", shuffle=True,
                                            batch_size=batch_size,
                                            num_workers=num_workers)
    val_loader = datamanager.get_dataloader(split="validation",
                                            batch_size=batch_size,
                                            num_workers=num_workers)
    pos_weight = 1 / datamanager.dataset["train"].target.mean() - 1
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, 
                                                           dtype=torch.float32,
                                                           device=model.device))
    logs = {"dataset": dataset, "label": "diagnosis"}
    if fold is not None:
        logs["fold"] = fold
    
    model.transfer(train_loader, val_loader, pretrained_epoch=pretrained_epoch, 
                   loss_fn=loss_fn, nb_epochs=nb_epochs, chkpt_dir=chkpt_dir, 
                   lr=lr, weight_decay=weight_decay,
                   refit=refit, logs=logs)
    
    train_loader = datamanager.get_dataloader(split="train", shuffle=False,
                                    batch_size=batch_size,
                                    num_workers=num_workers)
    internal_test_loader = datamanager.get_dataloader(split="internal_test",
                                                    batch_size=30,
                                                    num_workers=num_workers)
    external_test_loader = datamanager.get_dataloader(split="external_test",
                                                      batch_size=30,
                                                      num_workers=num_workers)
    metrics = {
        "roc_auc": lambda y_true, y_pred: roc_auc_score(y_true=y_true, y_score=y_pred),
        "balanced_accuracy": lambda y_true, y_pred: balanced_accuracy_score(y_true=y_true,
                                                                            y_pred=(y_pred > 0.5).astype(int))
    }
    model.test_classifier(loaders=[train_loader, val_loader,
                                   internal_test_loader, external_test_loader],
                          splits=["train", "validation", "internal_test", "external_test"],
                          epoch=(nb_epochs-1), metrics=metrics, chkpt_dir=chkpt_dir,
                          logs=logs)

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
    nb_folds = params.get("nb_folds")
    with open(os.path.join(chkpt_dir, "hyperparameters.json"), "r") as json_file:
        hyperparameters = json.load(json_file)
    n_embedding = hyperparameters.get("n_embedding", config.n_embedding)
    epoch_f = hyperparameters.get("nb_epochs", config.nb_epochs) - 1
    for dataset in ("asd", "bd", "scz"):
        logger.info(f"Dataset: {dataset}")
        if nb_folds is None:
            chkpt_dir_dt = os.path.join(chkpt_dir, dataset)
            os.makedirs(chkpt_dir_dt, exist_ok=True)
            if not os.path.islink(os.path.join(chkpt_dir_dt, f"barlowtwins_ep-{epoch_f}.pth")):
                os.symlink(os.path.join(chkpt_dir,
                                        f"barlowtwins_ep-{epoch_f}.pth"),
                        os.path.join(chkpt_dir_dt, f"barlowtwins_ep-{epoch_f}.pth"))
            
            fine_tune_bt_model(chkpt_dir_dt, dataset,
                               n_embedding=n_embedding, 
                               pretrained_epoch=epoch_f,
                               refit=params.get("refit", False),
                               nb_epochs=params.get("nb_epochs", config.nb_epochs_ft), 
                               lr=params.get("lr", config.lr_ft), 
                               weight_decay=params.get("weight_decay", config.weight_decay_ft),
                               batch_size=params.get("batch_size", config.batch_size), 
                               num_workers=params.get("num_workers", config.num_workers))
        else:
            for fold in range(nb_folds):
                logger.info(f"Fold: {fold}")
                chkpt_dir_dt = os.path.join(chkpt_dir, dataset, f"fold-{fold}")
                os.makedirs(chkpt_dir_dt, exist_ok=True)
                if not os.path.islink(os.path.join(chkpt_dir_dt, f"barlowtwins_ep-{epoch_f}.pth")):
                    os.symlink(os.path.join(chkpt_dir,
                                            f"barlowtwins_ep-{epoch_f}.pth"),
                            os.path.join(chkpt_dir_dt, f"barlowtwins_ep-{epoch_f}.pth"))
                
                fine_tune_bt_model(chkpt_dir_dt, dataset,
                                   n_embedding=n_embedding, 
                                   pretrained_epoch=epoch_f,
                                   fold=fold,
                                   nb_epochs=params.get("nb_epochs", config.nb_epochs_ft), 
                                   lr=params.get("lr", config.lr_ft), 
                                   weight_decay=params.get("weight_decay", config.weight_decay_ft),
                                   batch_size=params.get("batch_size", config.batch_size), 
                                   num_workers=params.get("num_workers", config.num_workers))

def nss_predictions(params):
    chkpt_dir = params["chkpt_dir"]
    nb_folds = 10
    with open(os.path.join(chkpt_dir, "hyperparameters.json"), "r") as json_file:
        hyperparameters = json.load(json_file)
    n_embedding = hyperparameters.get("n_embedding", config.n_embedding)
    #epoch_f = hyperparameters.get("nb_epochs", config.nb_epochs) - 1
    epoch_f = 299
    
    for fold in range(nb_folds):
        logger.info(f"Fold: {fold}")
        chkpt_dir_dt = os.path.join(chkpt_dir, "ausz", f"fold-{fold}")
        os.makedirs(chkpt_dir_dt, exist_ok=True)
        if not os.path.islink(os.path.join(chkpt_dir_dt, f"barlowtwins_ep-{epoch_f}.pth")):
            os.symlink(os.path.join(chkpt_dir,
                                    f"barlowtwins_ep-{epoch_f}.pth"),
                    os.path.join(chkpt_dir_dt, f"barlowtwins_ep-{epoch_f}.pth"))
        
        nb_epochs=params.get("nb_epochs", config.nb_epochs_ft)
        lr=params.get("lr", config.lr_ft)
        weight_decay=params.get("weight_decay", config.weight_decay_ft)
        batch_size=params.get("batch_size", config.batch_size)
        num_workers=params.get("num_workers", config.num_workers)
        model = BTModel(n_embedding=n_embedding, classifier=True)
    
        datamanager = DataManager(dataset="ausz", label="NSS", fold=fold,
                                  two_views=False, data_augmentation=None)  
    
        train_loader = datamanager.get_dataloader(split="train", shuffle=True,
                                                batch_size=batch_size,
                                                num_workers=num_workers)
        test_loader = datamanager.get_dataloader(split="test",
                                                batch_size=batch_size,
                                                num_workers=num_workers)
        loss_fn = nn.L1Loss()
        logs = {"dataset": "ausz", "label": "nss", "fold": fold}
        model.transfer(train_loader, test_loader, pretrained_epoch=epoch_f, 
                            loss_fn=loss_fn, nb_epochs=nb_epochs, chkpt_dir=chkpt_dir_dt, 
                            lr=lr, weight_decay=weight_decay,
                            logs=logs)
        
        train_loader = datamanager.get_dataloader(split="train", shuffle=False,
                                        batch_size=config.batch_size,
                                        num_workers=config.num_workers)
        metrics = {
            "r2_score": lambda y_true, y_pred: r2_score(y_true=y_true, y_pred=y_pred),
            "root_mean_squared_error": lambda y_true, y_pred: mean_squared_error(y_true=y_true,
                                                                                  y_pred=y_pred,
                                                                                  squared=False),
            "mean_absolute_error": lambda y_true, y_pred: mean_absolute_error(y_true=y_true,
                                                                              y_pred=y_pred)
        }
        model.test_classifier(loaders=[train_loader, test_loader],
                                splits=["train", "test"],
                                epoch=(nb_epochs-1), metrics=metrics, chkpt_dir=chkpt_dir_dt,
                                logs={"dataset": "ausz", "label": "nss", "fold": fold})

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--chkpt_dir", required=True, type=str,
    help="Checkpoint dir where all the logs are stored. List of existing checkpoint directories: " \
        + " - ".join(os.listdir(config.path2models)))
    parser.add_argument("-t", "--train", action="store_true", help="Train the model")
    parser.add_argument("-f", "--fine_tune", action="store_true", help="Finetune the model")
    parser.add_argument("-p", "--nss_predictions", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbosity")
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
    level = "debug" if args.verbose else "info"
    setup_logging(level=level, 
                  logfile=os.path.join(chkpt_dir, "train_bt_model.log"))
    logger = logging.getLogger("train")
    if args.train:
        train(params)
    if args.fine_tune:
        fine_tune(params)
    if args.nss_predictions:
        nss_predictions(params)
    