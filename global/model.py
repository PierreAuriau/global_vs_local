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
from loss import BarlowTwinsLoss
from log import TrainLogger

logging.setLoggerClass(TrainLogger)


class BTModel(nn.Module):

    def __init__(self, n_embedding=256):
        super().__init__()
        
        self.n_embedding = n_embedding
        self.encoder = densenet121(n_embedding=n_embedding, in_channels=1)
        self.projector = nn.Sequential(nn.Linear(n_embedding, 2 * n_embedding),
                                       nn.BatchNorm1d(2 * n_embedding),
                                       nn.ReLU(),
                                       nn.Linear(2 * n_embedding, n_embedding))
        self.logger = logging.getLogger("btmodel")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Device used : {self.device}")
        self = self.to(self.device)

    def forward(self, x):
        z = self.encoder(x) 
        proj = self.projector(z)
        return proj
    
    def configure_optimizers(self, **kwargs):
        return optim.Adam(self.parameters(), **kwargs)
    
    def fit(self, train_loader, val_loader, nb_epochs, 
            correlation, lambda_param,
            chkpt_dir, **kwargs_optimizer):
        
        self.optimizer = self.configure_optimizers(**kwargs_optimizer)
        self.lr_scheduler = None
        self.save_hyperparameters(chkpt_dir, {"lambda_param": lambda_param, 
                                              "correlation": correlation})
        self.logger.reset_history()
        self.scaler = GradScaler()
        self.loss_fn = BarlowTwinsLoss(correlation=correlation,
                                       lambda_param=lambda_param / float(self.n_embedding))
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
            self.logger.store({"epoch": epoch, "set": "train", "loss": train_loss})

            if epoch % 5 == 0:
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
                self.logger.store({"epoch": epoch, "set": "validation", "loss": val_loss})

            if epoch % 10 == 0:
                self.logger.info(f"Loss: train: {train_loss:.2g} / val: {val_loss:.2g}")
                self.logger.info(f"Training duration: {self.logger.get_duration()}")
                self.save_chkpt(os.path.join(chkpt_dir,
                                             f'barlowtwins_ep-{epoch}.pth'),
                                             save_optimizer=True)
                self.logger.save(chkpt_dir, filename="_train")
        
        self.logger.save(chkpt_dir, filename="_train")
        self.save_chkpt(os.path.join(chkpt_dir,
                                     f'barlowtwins_ep-{epoch}.pth'),
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
    
    def get_embeddings(self, loader):
        embeddings = []
        for batch in loader:
            x = batch["skeleton"].to(self.device) 
            z = self.encoder(x)
            embeddings.extend(z.cpu().numpy())
        return np.asarray(embeddings)

    def test(self, loader, epoch, preproc, label, chkpt_dir, save_y_pred=False):
        # FIXME
        """
        self.load_chkpt(os.path.join(chkpt_dir,
                                f'dlmodel_preproc-{preproc}_fold-{fold}_ep-{epoch}.pth'))
        test_loss = 0
        self.logger.reset_history()
        self.logger.step()
        y_pred, y_true = [], []
        for batch in tqdm(loader, desc="test"):
            inputs = batch[preproc].to(self.device)
            targets = batch[label].to(self.device)
            with torch.no_grad():
                outputs = self(inputs)
                loss = self.loss_fn(outputs, targets, label)
            test_loss += loss.item()
            y_pred.extend(outputs.squeeze().cpu().numpy())
            y_true.extend(targets.cpu().numpy())
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        self.logger.info(f"Test loss : {test_loss:.2g}")
        self.logger.reduce(reduce_fx="sum")
        self.logger.store({"fold": fold, "epoch": epoch, "set": "test", "loss": test_loss})

        r2 = r2_score(y_pred=y_pred, y_true=y_true)
        mae = mean_absolute_error(y_pred=y_pred, y_true=y_true)
        rmse = mean_squared_error(y_pred=y_pred, y_true=y_true, squared=False)
        self.logger.store({"r2": r2, "mean_absolute_error": mae, "root_mean_squarred_error": rmse})
        
        metrics = {"r2": r2_score,
                   "rmse": lambda y_pred, y_true: mean_squared_error(y_pred=y_pred, y_true=y_true, squared=False),
                   "mae": mean_absolute_error}
        for name, metric in metrics.items():
            value = metric(y_pred=y_pred, y_true=y_true)
            self.logger.store(**{name: value})
        
        self.logger.save(chkpt_dir, filename="_test")
        if save_y_pred:
            np.save(os.path.join(chkpt_dir, f"y_pred_fold-{fold}_epoch-{epoch}_test.npy"), y_pred)
            np.save(os.path.join(chkpt_dir, f"y_true_fold-{fold}_epoch-{epoch}_test.npy"), y_true)
        """

    def save_hyperparameters(self, chkpt_dir, hp={}):
        hp = {"n_embedding": self.n_embedding, **hp}
        with open(os.path.join(chkpt_dir, "hyperparameters.json"), "w") as f:
            json.dump(hp, f)

    def save_chkpt(self, filename, save_optimizer=False):
        torch.save({"encoder": self.encoder.state_dict(),
                    "projector": self.projector.state_dict()},
                   filename)
        if save_optimizer:
            torch.save({"optimizer": self.optimizer.state_dict()},
                        os.path.join(os.path.dirname(filename), f"optimizer.pth"))
    
    def load_chkpt(self, filename):
        chkpt = torch.load(filename, weights_only=True)
        status = self.encoder.load_state_dict(chkpt["encoder"], strict=False)
        self.logger.info(f"Loading encoder : {status}")
        status = self.projector.load_state_dict(chkpt["classifier"], strict=False)
        self.logger.info(f"Loading projector : {status}")

if __name__ == "__main__":
    model = BTModel()
    