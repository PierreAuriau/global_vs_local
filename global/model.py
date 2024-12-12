# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import json
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, \
                            mean_squared_error, r2_score, roc_auc_score

from densenet import densenet121
from mlp import Classifier, Projector
from loss import BarlowTwinsLoss
from log import TrainLogger

logging.setLoggerClass(TrainLogger)


class BTModel(nn.Module):

    def __init__(self, n_embedding=256, projector=False, classifier=False):
        super().__init__()
        
        self.n_embedding = n_embedding
        self.encoder = densenet121(n_embedding=n_embedding, in_channels=1)
        if projector:
            self.projector = Projector(latent_dim=n_embedding)
            self.logger = logging.getLogger("btmodel")
        else:
            self.projector = None
        if classifier:
            self.classifier = Classifier(latent_dim=n_embedding,
                                         activation="sigmoid")
            self.logger = logging.getLogger("classifier")
        else:
            self.classifier = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.logger.info(f"Device used : {self.device}")
        except AttributeError:
            self.logger = logging.getLogger("encoder")
            self.logger.info(f"Device used : {self.device}")
        self = self.to(self.device)

    def forward(self, x):
        z = self.encoder(x)
        if self.projector is not None:
            proj = self.projector(z)
            return proj
        if self.classifier is not None:
            out = self.classifier(z)
            return out
        return z
    
    def configure_optimizers(self, **kwargs):
        return optim.Adam(self.parameters(), **kwargs)
    
    def fit(self, train_loader, val_loader, nb_epochs, 
            correlation_bt, lambda_bt, chkpt_dir,
            logs={}, **kwargs_optimizer):
        
        self.optimizer = self.configure_optimizers(**kwargs_optimizer)
        self.lr_scheduler = None
        self.save_hyperparameters(chkpt_dir, {"lambda_bt": lambda_bt, 
                                              "correlation_bt": correlation_bt,
                                              "nb_epochs": nb_epochs,
                                              **kwargs_optimizer})
        self.logger.reset_history()
        self.scaler = GradScaler()
        self.loss_fn = BarlowTwinsLoss(correlation=correlation_bt,
                                       lambda_param=lambda_bt / float(self.n_embedding))
        for epoch in range(nb_epochs):
            self.logger.info(f"Epoch: {epoch}")
            # Training
            train_loss = 0
            self.train()
            self.logger.step()
            for batch in tqdm(train_loader, desc="train"):
                view_1 = batch["view_1"].to(self.device)
                view_2 = batch["view_2"].to(self.device)
                loss = self.training_step(view_1, view_2)
                train_loss += loss
            self.logger.reduce(reduce_fx="sum")
            self.logger.store({"epoch": epoch, "set": "train", "loss": train_loss, **logs})

            if epoch % 10 == 0:
                self.logger.info(f"Train loss: {train_loss:.2g}")
                if val_loader is not None:
                    # Validation
                    val_loss = 0
                    self.eval()
                    self.logger.step()
                    for batch in tqdm(val_loader, desc="Validation"):
                        view_1 = batch["view_1"].to(self.device)
                        view_2 = batch["view_2"].to(self.device)
                        loss = self.valid_step(view_1, view_2)
                        val_loss += loss
                    self.logger.reduce(reduce_fx="sum")
                    self.logger.store({"epoch": epoch, "set": "validation", "loss": val_loss, **logs})
                    self.logger.info(f"Validation loss: {val_loss:.2g}")
                self.logger.info(f"Training duration: {self.logger.get_duration()}")
                self.save_chkpt(chkpt_dir=chkpt_dir,
                                filename=f'barlowtwins_ep-{epoch}.pth',
                                save_optimizer=True)
                self.logger.save(chkpt_dir, filename="_train")
        
        self.logger.save(chkpt_dir, filename="_train")
        self.save_chkpt(chkpt_dir=chkpt_dir,
                        filename=f'barlowtwins_ep-{epoch}.pth',
                        save_optimizer=True)
        self.logger.info(f"End of training: {self.logger.get_duration()}")

    def training_step(self, view_1, view_2):
        self.optimizer.zero_grad()
        with autocast(device_type=self.device.type, dtype=torch.float16):
            zp_1 = self(view_1)
            zp_2 = self(view_2)
            loss = self.loss_fn(zp_1, zp_2)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.lr_scheduler:
            self.lr_scheduler.step()
        return loss.item()
    
    def valid_step(self, view_1, view_2):
        with torch.no_grad():
            zp_1 = self(view_1)
            zp_2 = self(view_2)
            loss = self.loss_fn(zp_1, zp_2)
        return loss.item()
    
    def get_embeddings(self, loader, label=False):
        embeddings = []
        labels = []
        for batch in tqdm(loader, desc="embeddings"):
            with torch.no_grad():
                x = batch["input"].to(self.device) 
                z = self.encoder(x)
            embeddings.extend(z.cpu().numpy())
            if label:
                labels.extend(batch["label"].numpy())
        if label:
            return np.asarray(embeddings), np.asarray(labels)
        return np.asarray(embeddings)
    
    def test_linear_probe(self, train_loader, val_loader, epochs, label, chkpt_dir, save_y_pred=False):
        self.logger.reset_history()
        for epoch in epochs:
            self.load_chkpt(chkpt_dir=chkpt_dir,
                            filename=f'barlowtwins_ep-{epoch}.pth')
            self.eval()
            z_train, y_train = self.get_embeddings(loader=train_loader, label=True)
            z_val, y_val = self.get_embeddings(loader=val_loader, label=True)
            clf = LogisticRegression(max_iter=10000)
            clf.fit(z_train, y_train)
            for split in ("train", "validation"):
                self.logger.step()
                if split == "train":
                    y_pred = clf.predict_proba(z_train)
                    y_true = y_train
                elif split == "validation":
                    y_pred = clf.predict_proba(z_val)
                    y_true = y_val
                self.logger.store({
                                "epoch": epoch,
                                "label": label,
                                "set": split,
                                "roc_auc": roc_auc_score(y_score=y_pred[:, 1], y_true=y_true),
                                "balanced_accuracy": balanced_accuracy_score(y_pred=y_pred.argmax(axis=1), y_true=y_true)})
            if save_y_pred:
                np.save(os.path.join(chkpt_dir, f"y_pred_label-{label}_epoch-{epoch}_set-{split}.npy"), y_pred)
        self.logger.save(chkpt_dir, filename="_test")

    def transfer(self, train_loader, val_loader,
                 pretrained_epoch, loss_fn, nb_epochs, 
                 chkpt_dir, refit=False, logs={}, **kwargs_optimizer):
        
        self.load_chkpt(chkpt_dir=chkpt_dir, 
                        filename=f'barlowtwins_ep-{pretrained_epoch}.pth')
        self.save_hyperparameters(chkpt_dir, {"pretrained_epoch": pretrained_epoch,
                                              "nb_epochs": nb_epochs,
                                              "loss": str(loss_fn),
                                              **kwargs_optimizer})
        if refit:
            self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.classifier.parameters()),
                                        **kwargs_optimizer)
        else:
            self.encoder.requires_grad_(False) # freeze encoder
            self.encoder.eval()
            self.optimizer = optim.Adam(self.classifier.parameters(), **kwargs_optimizer)
        self.lr_scheduler = None
        self.scaler = GradScaler()
        self.logger.reset_history()
        self.loss_fn = loss_fn
        for epoch in range(nb_epochs):
            self.logger.info(f"Epoch: {epoch}")
            # Training
            train_loss = 0
            self.classifier.train()
            if refit:
                self.encoder.train()
            self.logger.step()
            for batch in tqdm(train_loader, desc="train"):
                input = batch["input"].to(self.device)
                label = batch["label"].to(self.device)
                loss = self.transfer_step(input, label)
                train_loss += loss
            self.logger.reduce(reduce_fx="sum")
            self.logger.store({"epoch": epoch, "set": "train", "loss": train_loss, **logs})
            
            # Validation
            if val_loader is not None:
                val_loss = 0
                self.classifier.eval()
                self.encoder.eval()
                self.logger.step()
                for batch in tqdm(val_loader, desc="Validation"):
                    input = batch["input"].to(self.device)
                    label = batch["label"].to(self.device)
                    loss = self.valid_classifier_step(input, label)
                    val_loss += loss
                self.logger.reduce(reduce_fx="sum")
                self.logger.store({"epoch": epoch, "set": "validation", "loss": val_loss, **logs})

            if epoch % 10 == 0:
                self.logger.info(f"Loss: train: {train_loss:.2g} / val: {val_loss:.2g}")
                self.logger.info(f"Training duration: {self.logger.get_duration()}")
                self.save_chkpt(chkpt_dir=chkpt_dir, 
                                filename=f'classifier_ep-{epoch}.pth')
        
        self.logger.save(chkpt_dir, filename="_train")
        self.save_chkpt(chkpt_dir=chkpt_dir, 
                        filename=f'classifier_ep-{epoch}.pth')
        self.logger.info(f"End of training: {self.logger.get_duration()}")
    
    def transfer_step(self, input, label):
        self.optimizer.zero_grad()
        with autocast(device_type=self.device.type, dtype=torch.float16):
            z = self.encoder(input)
            pred = self.classifier(z, return_logits=True)
            loss = self.loss_fn(pred.squeeze(), label)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.lr_scheduler:
            self.lr_scheduler.step()
        return loss.item()

    def valid_classifier_step(self, input, label):
        with torch.no_grad():
            z = self.encoder(input)
            pred = self.classifier(z, return_logits=True)
            loss = self.loss_fn(pred.squeeze(), label)
        return loss.item()
    
    def test_classifier(self, loaders, splits,
                        chkpt_dir, epoch, metrics, logs={}):
        self.logger.info("Testing predictor")
        self.metrics = metrics
        self.logger.reset_history()
        self.load_chkpt(chkpt_dir=chkpt_dir,
                        filename=f"classifier_ep-{epoch}.pth")
        self.eval()
        for split, loader in zip(splits, loaders):
            self.logger.step()
            values = self.test_classifier_step(loader=loader,
                                                split=split)
            self.logger.store({"epoch": epoch,
                               "set": split,
                               **values,
                               **logs})
        self.logger.save(chkpt_dir, filename="_test")
    
    def test_classifier_step(self, loader, split):
        y_true = []
        y_pred = []
        for batch in tqdm(loader, desc=split):
            input = batch["input"].to(self.device)
            with torch.no_grad():
                z = self.encoder(input)
                pred = self.classifier(z)
            y_pred.extend(pred.cpu().numpy())
            y_true.extend(batch["label"].numpy())
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        values = {}
        for name, metric in self.metrics.items():
            values[name] = metric(y_true=y_true,
                                  y_pred=y_pred)
        return values
        
    def save_hyperparameters(self, chkpt_dir, hp={}):
        hp = {"n_embedding": self.n_embedding, **hp}
        with open(os.path.join(chkpt_dir, "hyperparameters.json"), "w") as f:
            json.dump(hp, f)

    def save_chkpt(self, chkpt_dir, filename, save_optimizer=False):
        to_save = {"encoder": self.encoder.state_dict()}
        if self.classifier is not None:
            to_save["classifier"] = self.classifier.state_dict()
        if self.projector is not None:
            to_save["projector"] = self.projector.state_dict()
        torch.save(to_save,
                   os.path.join(chkpt_dir, filename))
        if save_optimizer:
            torch.save(self.optimizer.state_dict(),
                       os.path.join(chkpt_dir, "optimizer.pth"))
    
    def load_chkpt(self, chkpt_dir, filename, load_optimizer=False):
        chkpt = torch.load(os.path.join(chkpt_dir, filename), weights_only=True)
        status = self.encoder.load_state_dict(chkpt["encoder"], strict=False)
        self.logger.info(f"Loading encoder : {status}")
        if (self.projector is not None) & ("projector" in chkpt.keys()):
            status = self.projector.load_state_dict(chkpt["projector"], strict=False)
            self.logger.info(f"Loading projector : {status}")
        if ("classifier" in chkpt.keys()) & (self.classifier is not None):
            status = self.classifier.load_state_dict(chkpt["classifier"], strict=False)
            self.logger.info(f"Loading classifier : {status}")
        if load_optimizer:
            self.optimizer.load_state_dict(torch.load(os.path.join(chkpt_dir, "optimizer.pth"),
                                                      weights_only=True))

if __name__ == "__main__":
    model = BTModel()
    