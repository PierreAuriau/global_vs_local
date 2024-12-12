# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re
import json
import argparse

import pandas as pd
import numpy as np
from umap import UMAP
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, \
                            mean_squared_error, r2_score, roc_auc_score
import matplotlib.pyplot as plt

# from project
from log import setup_logging
from model import BTModel
from datamanager import DataManager
from config import Config

config = Config()

def test_bt_model(chkpt_dir, label="sex"):
    with open(os.path.join(chkpt_dir, "hyperparameters.json"), "r") as json_file:
        hyperparameters = json.load(json_file)
    n_embedding = hyperparameters.get("n_embedding", config.n_embedding)
    nb_epochs = hyperparameters.get("nb_epochs", config.nb_epochs)
    epochs_to_test = [i for i in range(0, nb_epochs, 10)] + [nb_epochs-1]
    model = BTModel(n_embedding=n_embedding, projector=False)
    datamanager = DataManager(dataset="ukb", label=label, two_views=False,  
                              data_augmentation=None)
    
    train_loader = datamanager.get_dataloader(split="train", shuffle=False,
                                              batch_size=config.batch_size,
                                              num_workers=config.num_workers)
    val_loader = datamanager.get_dataloader(split="validation",
                                            batch_size=config.batch_size,
                                            num_workers=config.num_workers)
    
    model.test_linear_probe(train_loader=train_loader, val_loader=val_loader,
                            label=label, epochs=epochs_to_test, 
                            chkpt_dir=chkpt_dir)
    
def test_bt_model_dx(chkpt_dir):
    with open(os.path.join(chkpt_dir, "hyperparameters.json"), "r") as json_file:
        hyperparameters = json.load(json_file)
    n_embedding = hyperparameters.get("n_embedding", config.n_embedding)
    nb_epochs = hyperparameters.get("nb_epochs", config.nb_epochs)
    epochs_to_test = [i for i in range(0, nb_epochs, 10)] + [nb_epochs-1]
    
    model = BTModel(n_embedding=n_embedding, projector=False)

    for dataset in ("asd", "bd", "scz"):
        datamanager = DataManager(dataset=dataset, label="diagnosis", two_views=False,  
                                  data_augmentation=None)
        chkpt_dir_dt = os.path.join(chkpt_dir, dataset)
        os.makedirs(chkpt_dir_dt, exist_ok=True)
        
        to_unlink = []
        for epoch in epochs_to_test:
            if not os.path.exists(os.path.join(chkpt_dir_dt, 
                                               f"barlowtwins_ep-{epoch}.pth")):
                os.symlink(os.path.join(chkpt_dir,
                                        f"barlowtwins_ep-{epoch}.pth"),
                        os.path.join(chkpt_dir_dt, f"barlowtwins_ep-{epoch}.pth"))
                to_unlink.append(os.path.join(chkpt_dir_dt, f"barlowtwins_ep-{epoch}.pth"))
        train_loader = datamanager.get_dataloader(split="train", shuffle=False,
                                                batch_size=config.batch_size,
                                                num_workers=config.num_workers)
        val_loader = datamanager.get_dataloader(split="validation",
                                                batch_size=config.batch_size,
                                                num_workers=config.num_workers)
    
        model.test_linear_probe(train_loader=train_loader, val_loader=val_loader,
                                label="diagnosis",
                                epochs=epochs_to_test, 
                                chkpt_dir=chkpt_dir_dt)
        for link in to_unlink:
            os.unlink(link)

def test_classifier(chkpt_dir, epoch=-1):
    with open(os.path.join(chkpt_dir, "hyperparameters.json"), "r") as json_file:
        hyperparameters = json.load(json_file)
    n_embedding = hyperparameters.get("n_embedding", config.n_embedding)
    model = BTModel(n_embedding=n_embedding, classifier=True)
    for dataset in ("asd", "bd", "scz"):
        chkpt_dir_dt = os.path.join(chkpt_dir, dataset)
        with open(os.path.join(chkpt_dir_dt, "hyperparameters.json"), "r") as json_file:
            hyperparameters = json.load(json_file)
        if epoch < 0:
            epoch = hyperparameters.get("nb_epochs", config.nb_epochs_ft) + epoch
        datamanager = DataManager(dataset=dataset, label="diagnosis", two_views=False,  
                                data_augmentation=None)
        loaders = [datamanager.get_dataloader(split=s, 
                                            batch_size=config.batch_size, 
                                            num_workers=config.num_workers) 
                    for s in config.splits]
        metrics = {
        "roc_auc": lambda y_true, y_pred: roc_auc_score(y_true=y_true, y_score=y_pred),
        "balanced_accuracy": lambda y_true, y_pred: balanced_accuracy_score( y_true=y_true,
                                                                            y_pred=y_pred.argmax(axis=1))
                                                                            }
        
        model.test_classifier(loaders=loaders, splits=config.splits, 
                              epoch=epoch, metrics=metrics, chkpt_dir=chkpt_dir_dt,
                              logs={"dataset": dataset, "label": "diagnosis"})
    
def reduce_latent_space(chkpt_dir, pretrained_epoch=-1,
                           with_metadata=False):
    with open(os.path.join(chkpt_dir, "hyperparameters.json"), "r") as json_file:
        hyperparameters = json.load(json_file)
    n_embedding = hyperparameters.get("n_embedding", config.n_embedding)
    if pretrained_epoch < 0 :
        pretrained_epoch = hyperparameters.get("nb_epochs", config.nb_epochs) + pretrained_epoch
    if with_metadata:
        scheme = pd.read_csv(os.path.join(config.path2schemes, ""))
        metadata = pd.read_csv(os.path.join(config.path2data, ""))
        assert (metadata["participant_id"] == scheme["participant_id"]).all()
    model = BTModel(n_embedding=n_embedding)
    model.load_chkpt(chkpt_dir=chkpt_dir,
                     filename=f'barlowtwins_ep-{pretrained_epoch}.pth')
    model.eval()
    datamanager = DataManager(dataset="ukb", label=None, two_views=False,  
                              data_augmentation=None)
    
    train_loader = datamanager.get_dataloader(split="train",
                                              batch_size=config.batch_size,
                                            num_workers=config.num_workers)
    val_loader = datamanager.get_dataloader(split="validation", 
                                            batch_size=config.batch_size,
                                            num_workers=config.num_workers)
    z_train = model.get_embeddings(train_loader)
    z_val = model.get_embeddings(val_loader)
    np.save(os.path.join(chkpt_dir, f"embeddings_ep-{pretrained_epoch}_set-train.npy"),
        z_train.astype(np.float32))
    np.save(os.path.join(chkpt_dir, f"embeddings_ep-{pretrained_epoch}_set-validation.npy"),
        z_val.astype(np.float32))    

    reducer = make_pipeline(StandardScaler(),
                            UMAP(n_neighbors=15,
                                    min_dist=1.0,
                                    n_components=2))
    z_red_train = reducer.fit_transform(z_train)
    z_red_val = reducer.transform(z_val)

    np.save(os.path.join(chkpt_dir, f"embeddings_reduced_ep-{pretrained_epoch}_set-train.npy"),
        z_red_train.astype(np.float32))
    np.save(os.path.join(chkpt_dir, f"embeddings_reduced_ep-{pretrained_epoch}_set-validation.npy"),
        z_red_val.astype(np.float32))
    
    if with_metadata:
        metadata.loc[scheme["set"] == "train", ["umap_1", "umap_2"]] = z_red_train
        metadata.loc[scheme["set"] == "validation", ["umap_1", "umap_2"]] = z_red_val
        metadata.to_csv(os.path.join(chkpt_dir, f"participants_umap_ep-{pretrained_epoch}.csv"),
                        sep=",", index=False)

    for dataset in ("asd", "bd", "scz"):
        datamanager = DataManager(dataset=dataset, label=None,  
                                  data_augmentation=None)

        train_loader = datamanager.get_dataloader(split="train",
                                                  batch_size=config.batch_size,
                                                  num_workers=config.num_workers)
        val_loader = datamanager.get_dataloader(split="validation", 
                                                batch_size=config.batch_size,
                                                num_workers=config.num_workers)
        test_int_loader = datamanager.get_dataloader(split="internal_test",
                                                       batch_size=config.batch_size,
                                                       num_workers=config.num_workers)
        test_ext_loader = datamanager.get_dataloader(split="external_test",
                                                 batch_size=config.batch_size,
                                                 num_workers=config.num_workers)
            
        for split, loader in zip(["train", "validation", "internal_test", "external_test"],
                                 [train_loader, val_loader, test_int_loader, test_ext_loader]
                                 ):
            z = model.get_embeddings(loader)
            z_red = reducer.transform(z)
            os.makedirs(os.path.join(chkpt_dir, dataset), exist_ok=True)
            np.save(os.path.join(chkpt_dir, dataset, f"embeddings_ep-{pretrained_epoch}_set-{split}.npy"),
                    z.astype(np.float32))
            np.save(os.path.join(chkpt_dir, dataset, f"embeddings_reduced_ep-{pretrained_epoch}_set-{split}.npy"),
                    z_red.astype(np.float32))
    

    
def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--chkpt_dir", required=True, type=str,
    help="Checkpoint dir where all the logs are stored. List of existing checkpoint directories: " \
        + " - ".join(os.listdir(config.path2models)))
    parser.add_argument("--test_bt_model", action="store_true", help="")
    parser.add_argument("--test_bt_model_dx", action="store_true", help="")
    parser.add_argument("--test_classifier", action="store_true", help="")
    parser.add_argument("--reduce_latent_space", action="store_true", help="")
    args, unknownargs = parser.parse_known_args(argv)
    params = {}
    for i in range(0, len(unknownargs), 2):
        key = re.search("--([a-z_]+)", unknownargs[i])[1]
        try:
            params[key] = eval(unknownargs[i+1])
        except NameError:
            params[key] = unknownargs[i+1]
    return args, params


if __name__ == "__main__":

    args, params = parse_args(sys.argv[1:])
    chkpt_dir = os.path.join(config.path2models, args.chkpt_dir)
    assert os.path.exists(chkpt_dir), f"Checkpoint directory does not exist : {chkpt_dir}"
    setup_logging(level="info", 
                  logfile=os.path.join(chkpt_dir, "test_bt_model.log"))
    
    if args.test_bt_model:
        test_bt_model(chkpt_dir=chkpt_dir,
                      label=params.get("label", "sex"))
    if args.test_bt_model_dx:
        test_bt_model_dx(chkpt_dir=chkpt_dir)
    if args.test_classifier:
        test_classifier(chkpt_dir=chkpt_dir,
                        epoch=params.get("epoch", -1))
    if args.reduce_latent_space:
        reduce_latent_space(chkpt_dir=chkpt_dir,
                            pretrained_epoch=params.get("pretrained_epoch", -1))




