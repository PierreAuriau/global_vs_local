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
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, \
                            mean_squared_error, r2_score, roc_auc_score

from classifier import Classifier
from log import TrainLogger
from config import Config

logging.setLoggerClass(TrainLogger)
config = Config()


class DLModel(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()
        self.classifier = Classifier(latent_dim=latent_dim,
                                     activation="sigmoid")
        self.logger = logging.getLogger("dlmodel")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Device used : {self.device}")
        self = self.to(self.device)

    def forward(self, x):
        return self.classifier(x)
    
    def configure_optimizers(self, **kwargs):
        return optim.Adam(self.parameters(), **kwargs)

    def fit(self, train_loader, val_loader, nb_epochs, chkpt_dir, 
            logs={}, **kwargs_optimizer):
        
        self.optimizer = optim.Adam(self.classifier.parameters(), 
                                    **kwargs_optimizer)
        self.lr_scheduler = None
        self.scaler = GradScaler()
        self.logger.reset_history()
        self.save_hyperparameters(chkpt_dir=chkpt_dir,
                                  hp=kwargs_optimizer)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1/train_loader.dataset.target.mean() - 1, 
                                                                    dtype=torch.float32,
                                                                    device=self.device))
        for epoch in range(nb_epochs):
            self.logger.info(f"Epoch: {epoch}")
            # Training
            train_loss = 0
            self.classifier.train()
            self.logger.step()
            for batch in tqdm(train_loader, desc="train"):
                input = batch["input"].to(self.device)
                label = batch["label"].to(self.device)
                loss = self.training_step(input, label)
                train_loss += loss
            self.logger.reduce(reduce_fx="sum")
            self.logger.store({"epoch": epoch, "set": "train", 
                               "loss": train_loss, **logs})
            
            # Validation
            val_loss = 0
            self.classifier.eval()
            self.logger.step()
            for batch in tqdm(val_loader, desc="validation"):
                input = batch["input"].to(self.device)
                label = batch["label"].to(self.device)
                loss = self.valid_step(input, label)
                val_loss += loss
            self.logger.reduce(reduce_fx="sum")
            self.logger.store({"epoch": epoch, "set": "validation", 
                               "loss": val_loss, **logs})

            if epoch % 10 == 0:
                self.logger.info(f"Loss: train: {train_loss:.2g} / val: {val_loss:.2g}")
                self.logger.info(f"Training duration: {self.logger.get_duration()}")
                self.save_chkpt(chkpt_dir=chkpt_dir, 
                                filename=f'classifier_ep-{epoch}.pth')
        
        self.logger.save(chkpt_dir, filename="_train")
        self.save_chkpt(chkpt_dir=chkpt_dir, 
                        filename=f'classifier_ep-{epoch}.pth')
        self.logger.info(f"End of training: {self.logger.get_duration()}")
    
    def training_step(self, input, label):
        self.optimizer.zero_grad()
        with autocast(device_type=self.device.type, dtype=torch.float16):
            pred = self.classifier(input, return_logits=True)
            loss = self.loss_fn(pred.squeeze(), label)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.lr_scheduler:
            self.lr_scheduler.step()
        return loss.item()

    def valid_step(self, input, label):
        with torch.no_grad():
            pred = self.classifier(input, return_logits=True)
            loss = self.loss_fn(pred.squeeze(), label)
        return loss.item()
    
    def test(self, loaders, splits, chkpt_dir, epoch,
             save_y_pred=False, logs={}):
        self.logger.reset_history()
        self.load_chkpt(chkpt_dir=chkpt_dir,
                        filename=f"classifier_ep-{epoch}.pth")
        self.eval()
        for split, loader in zip(splits, loaders):
            self.logger.step()
            if save_y_pred:
                test_logs, y_pred, y_true = self.test_step(loader=loader, split=split, 
                                              return_y_pred=True)
                np.save(os.path.join(chkpt_dir, f"y_pred_ep-{epoch}_set-{split}.npy"),
                        y_pred.astype(np.float32))
                np.save(os.path.join(chkpt_dir, f"y_true_ep-{epoch}_set-{split}.npy"),
                        y_true.astype(np.float32))
            else:
                test_logs = self.test_step(loader=loader, split=split)
            self.logger.store({"epoch": epoch,
                               **logs,
                               **test_logs})
        self.logger.save(chkpt_dir, filename="_test")
    
    def test_step(self, loader, split, return_y_pred=False):
        y_true = []
        y_pred = []
        for batch in tqdm(loader, desc=split):
            input = batch["input"].to(self.device)
            with torch.no_grad():
                pred = self.classifier(input)
            y_pred.extend(pred.cpu().numpy())
            y_true.extend(batch["label"].numpy())
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        logs = {
            "set": split,
            "roc_auc": roc_auc_score(y_true=y_true, y_score=y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true=y_true, 
                                        y_pred=(y_pred > 0.5).astype(int))
               }
        if return_y_pred:
            return logs, y_pred, y_true
        return logs
    
    def test_linear_probe(self, predictions, labels, epoch,
                          chkpt_dir, logs={}, save_y_pred=False):
        
        # FIXME : add loadings ?
        """
        self.logger.reset_history()
        clf = LogisticRegression(max_iter=1000, C=1.0, penalty="l2", 
                                 fit_intercept=True)
        clf.fit(predictions["train"], labels["train"])
        
        cv = [(range(len(labels["train"])),
                                   range(len(labels["train"]), 
                                         len(labels["train"]) + len(labels["validation"])))]
        """
        cv = PredefinedSplit([-1 for _ in range(len(labels["train"]))] + \
                             [0 for _ in range(len(labels["validation"]))])
        clf = GridSearchCV(LogisticRegression(max_iter=1000),
                           param_grid={"C": 10. ** np.arange(-1, 3)},
                           cv=cv, 
                           n_jobs=config.num_workers)
        X = np.concatenate([predictions["train"], predictions["validation"]], axis=0)
        y = np.concatenate([labels["train"], labels["validation"]], axis=0)
        clf.fit(X, y)
        for split in predictions.keys():
            self.logger.step()
            y_pred = clf.predict_proba(predictions[split])
            y_true = labels[split]
            self.logger.store({
                            "epoch": epoch,
                            "set": split,
                            "roc_auc": roc_auc_score(y_score=y_pred[:, 1], y_true=y_true),
                            "balanced_accuracy": balanced_accuracy_score(y_pred=y_pred.argmax(axis=1), y_true=y_true),
                            **logs})
            if save_y_pred:
                np.save(os.path.join(chkpt_dir, f"y_pred_epoch-{epoch}_set-{split}.npy"), y_pred)
                np.save(os.path.join(chkpt_dir, f"y_true_epoch-{epoch}_set-{split}.npy"), y_true)
        if save_y_pred:
            # np.save(os.path.join(chkpt_dir, f"coef_epoch-{epoch}.npy"), clf.coef_)
            np.save(os.path.join(chkpt_dir, f"coef_epoch-{epoch}.npy"), clf.best_estimator_.coef_)
        self.logger.save(chkpt_dir, filename="_test")
        
    def save_hyperparameters(self, chkpt_dir, hp={}):
        with open(os.path.join(chkpt_dir, "hyperparameters.json"), "w") as f:
            json.dump(hp, f)

    def save_chkpt(self, chkpt_dir, filename, save_optimizer=False):
        to_save = {"classifier": self.classifier.state_dict()}
        torch.save(to_save,
                   os.path.join(chkpt_dir, filename))
        if save_optimizer:
            torch.save(self.optimizer.state_dict(),
                       os.path.join(chkpt_dir, "optimizer.pth"))
    
    def load_chkpt(self, chkpt_dir, filename, load_optimizer=False):
        chkpt = torch.load(os.path.join(chkpt_dir, filename), weights_only=True)
        status = self.classifier.load_state_dict(chkpt["classifier"], strict=False)
        self.logger.info(f"Loading classifier : {status}")
        if load_optimizer:
            self.optimizer.load_state_dict(torch.load(os.path.join(chkpt_dir, "optimizer.pth"),
                                                      weights_only=True))

if __name__ == "__main__":
    model = DLModel()
    