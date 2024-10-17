# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os

# from project
from log import setup_logging
from model import BTModel
from datamanager import DataManager
from config import Config

config = Config()


def train_bt_model():
    
    chkpt_dir = os.path.join(config.path2models, "20241015_global_ukb_bt_pretraining")
    os.makedirs(chkpt_dir, exist_ok=True)
    setup_logging(level="info", 
                  logfile=os.path.join(chkpt_dir, "train_bt_model.log"))
    model = BTModel(n_embedding=256)
    datamanager = DataManager(label=None, two_views=True,  
                              batch_size=32, data_augmentation="cutout",
                              num_workers=8, pin_memory=True)
    
    train_loader = datamanager.get_dataloader(split="train")
    val_loader = datamanager.get_dataloader(split="validation")
    
    model.fit(train_loader=train_loader, val_loader=val_loader, 
              nb_epochs=100,
              correlation=config.correlation_BT, lambda_param=config.lambda_BT,
              chkpt_dir=chkpt_dir)

if __name__ == "__main__":
    train_bt_model()