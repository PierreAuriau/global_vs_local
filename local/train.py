# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

# from project
from log import setup_logging
from model import DLModel
from datamanager import DataManager
from config import Config

config = Config()


logger = logging.getLogger("train")
   
def train_dl_model(chkpt_dir, dataset, area, reduced, fold):
        
    model = DLModel(latent_dim=config.n_components if reduced else config.latent_dim)

    datamanager = DataManager(dataset=dataset, area=area,
                              label="diagnosis", reduced=reduced, fold=fold)
    train_loader = datamanager.get_dataloader(split="train",
                                              batch_size=config.batch_size,
                                              shuffle=True, 
                                              num_workers=config.num_workers)
    val_loader = datamanager.get_dataloader(split="validation",
                                            batch_size=60,
                                            num_workers=config.num_workers)
    # training
    model.fit(train_loader, val_loader,
              nb_epochs=config.nb_epochs, chkpt_dir=chkpt_dir,
              logs={"area": area, "dataset": dataset, 
                    "fold": fold, "reduced": reduced},
              lr=config.lr, weight_decay=config.weight_decay)
    
    train_loader = datamanager.get_dataloader(split="train",
                                              batch_size=60,
                                              shuffle=False,
                                              num_workers=config.num_workers)
    test_int_loader = datamanager.get_dataloader(split="internal_test",
                                                batch_size=60,
                                                num_workers=config.num_workers)
    test_ext_loader = datamanager.get_dataloader(split="external_test",
                                             batch_size=60,
                                             num_workers=config.num_workers)
    # testing
    model.test(loaders=[train_loader, val_loader,
                        test_int_loader, test_ext_loader],
                splits=["train", "validation", "internal_test", "external_test"],
                epoch=config.epoch_f, chkpt_dir=chkpt_dir, save_y_pred=True,
                logs={"area": area, "dataset": dataset, 
                      "reduced": reduced, "fold": fold})

def test_linear_probe(chkpt_dir, dataset, reduced=False, fold=None):
    logger.info("Logistic Regression fitting")
    logs = defaultdict(list)   
    predictions = {}
    labels = {}
    # Data loading
    # for split in config.splits:
    for split in config.splits:
        predictions[split] = np.stack([
            np.load(os.path.join(chkpt_dir,
                                area,
                                f"y_pred_ep-{config.epoch_f}_set-{split}.npy"))
            for area in config.areas], 
            axis=1)
        labels[split] = np.stack([
            np.load(os.path.join(chkpt_dir,
                                area,
                                f"y_true_ep-{config.epoch_f}_set-{split}.npy"))
            for area in config.areas], 
            axis=1)
        # sanity check
        assert np.all(labels[split].transpose() == labels[split].transpose()[0])
        labels[split] = labels[split][:, 0]

    # Fit Logistic Regression
    clf = LogisticRegression(max_iter=1000, C=1.0, penalty="l2", 
                             fit_intercept=True)
    clf.fit(predictions["train"], labels["train"])

    # Test model
    # for split in config.splits:
    for split in config.splits:
        y_pred = clf.predict_proba(predictions[split])
        y_true = labels[split]
        logs["epoch"].append(config.epoch_f)
        logs["set"].append(split)
        logs["label"].append("diagnosis")
        logs["dataset"].append(dataset)
        logs["reduced"].append(reduced)
        logs["fold"].append(fold)
        logs["roc_auc"].append(roc_auc_score(y_score=y_pred[:, 1], y_true=y_true))
        logs["balanced_accuracy"].append(balanced_accuracy_score(y_pred=y_pred.argmax(axis=1), y_true=y_true))

    np.save(os.path.join(chkpt_dir, f"lrl2_epoch-{config.epoch_f}_coef_.npy"), clf.coef_)
    df_logs = pd.DataFrame(logs)
    df_logs.to_csv(os.path.join(chkpt_dir,
                                f"lrl2_epoch-{config.epoch_f}_test.csv"),
                    sep=",", index=False)

def test_linear_probe_cv(chkpt_dir, dataset, reduced=False, fold=None):
    logger.info("Logistic Regression CV fitting")
    predictions = {}
    labels = {}
    logs = defaultdict(list)
    # Load Data
    for split in config.splits:
        predictions[split] = np.stack([
            np.load(os.path.join(chkpt_dir,
                                area,
                                f"y_pred_ep-{config.epoch_f}_set-{split}.npy"))
            for area in config.areas], 
            axis=1)
        labels[split] = np.stack([
            np.load(os.path.join(chkpt_dir,
                                area,
                                f"y_true_ep-{config.epoch_f}_set-{split}.npy"))
            for area in config.areas], 
            axis=1)
        # sanity check
        assert np.all(labels[split].transpose() == labels[split].transpose()[0])
        labels[split] = labels[split][:, 0]
    # Fit Logistic Regression
    cv = PredefinedSplit([-1 for _ in range(len(labels["train"]))] + \
                            [0 for _ in range(len(labels["validation"]))])
    clf = GridSearchCV(LogisticRegression(max_iter=1000, penalty="l2"),
                        param_grid={"C": 10. ** np.arange(-1, 3)},
                        cv=cv, 
                        n_jobs=config.num_workers)
    X = np.concatenate([predictions["train"], predictions["validation"]], axis=0)
    y = np.concatenate([labels["train"], labels["validation"]], axis=0)
    clf.fit(X, y)
    logger.info(f"Best score: {clf.best_score_}")
    logger.info(f"Best params: {clf.best_params_}")
    # Test model
    for split in config.splits:
        y_pred = clf.predict_proba(predictions[split])
        y_true = labels[split]
        logs["epoch"].append(config.epoch_f)
        logs["set"].append(split)
        logs["label"].append("diagnosis")
        logs["dataset"].append(dataset)
        logs["reduced"].append(reduced)
        logs["fold"].append(fold)
        logs["score"].append(clf.best_score_)
        for param, value in clf.best_params_.items():
            logs[param].append(value)
        logs["roc_auc"].append(roc_auc_score(y_score=y_pred[:, 1], y_true=y_true))
        logs["balanced_accuracy"].append(balanced_accuracy_score(y_pred=y_pred.argmax(axis=1), y_true=y_true))

    np.save(os.path.join(chkpt_dir, f"lrl2_cv_epoch-{config.epoch_f}_coef_.npy"), clf.best_estimator_.coef_)
    df_logs = pd.DataFrame(logs)
    df_logs.to_csv(os.path.join(chkpt_dir,
                                f"lrl2_cv_epoch-{config.epoch_f}_test.csv"),
                    sep=",", index=False)

def train_all_dl_models(chkpt_dir, reduced=False):
    chkpt_dir = os.path.join(config.path2models, chkpt_dir)
    os.makedirs(chkpt_dir, exist_ok=True)
    setup_logging(level="info", 
                  logfile=os.path.join(chkpt_dir, "train_dl_model.log"))
    for dataset in config.datasets:
        logger.info(f"\n# DATASET: {dataset}\n" + "-"*(len(dataset)+11))
        for area in config.areas:
            logger.info(f"\n## AREA: {area}\n"+ "-"*(len(area)+10))
            chkpt_dir_da = os.path.join(chkpt_dir, dataset, area)
            os.makedirs(chkpt_dir_da, exist_ok=True)
            train_dl_model(chkpt_dir=chkpt_dir_da,
                           dataset=dataset,
                           area=area,
                           reduced=reduced)
        # Train logistic regressions
        logger.info(f"Training LogisticRegression")
        predictions = {}
        labels = {}
        logs = defaultdict(list)
        # Load Data
        for split in config.splits:
            predictions[split] = np.stack([
                np.load(os.path.join(chkpt_dir_da,
                                    area,
                                    f"y_pred_ep-{config.epoch_f}_set-{split}.npy"))
                for area in config.areas], 
                axis=1)
            labels[split] = np.stack([
                np.load(os.path.join(chkpt_dir_da,
                                    area,
                                    f"y_true_ep-{config.epoch_f}_set-{split}.npy"))
                for area in config.areas], 
                axis=1)
            # sanity check
            assert np.all(labels[split].transpose() == labels[split].transpose()[0])
            labels[split] = labels[split][:, 0]
        # Fit Logistic Regression
        clf = LogisticRegression(max_iter=1000, C=1.0, penalty="l2", 
                                    fit_intercept=True)
        clf.fit(predictions["train"], labels["train"])
        # Test model
        for split in config.splits:
            y_pred = clf.predict_proba(predictions[split])
            y_true = labels[split]
            logs["epoch"].append(config.epoch_f)
            logs["set"].append(split)
            logs["label"].append("diagnosis")
            logs["dataset"].append(dataset)
            logs["reduced"].append(reduced)
            logs["roc_auc"].append(roc_auc_score(y_score=y_pred[:, 1], y_true=y_true))
            logs["balanced_accuracy"].append(balanced_accuracy_score(y_pred=y_pred.argmax(axis=1), y_true=y_true))

        np.save(os.path.join(chkpt_dir_da, f"lrl2_epoch-{config.epoch_f}_coef_.npy"), clf.best_estimator_.coef_)
        df_logs = pd.DataFrame(logs)
        df_logs.to_csv(os.path.join(chkpt_dir_da,
                                    f"lrl2_epoch-{config.epoch_f}_test.csv"),
                        sep=",", index=False)

def train_all_dl_models_cv(chkpt_dir, dataset):
    chkpt_dir = os.path.join(config.path2models, chkpt_dir)
    os.makedirs(chkpt_dir, exist_ok=True)
    setup_logging(level="info", 
                  logfile=os.path.join(chkpt_dir, "train_dl_model.log"))
    # for dataset in config.datasets:
    logger.info(f"\n# DATASET: {dataset}\n" + "-"*(len(dataset)+11))
    for fold in range(config.nb_folds):
        logger.info(f"\n## FOLD: {fold}\n"+ "-"*11)
        if fold == 0:
            test_linear_probe(chkpt_dir=os.path.join(chkpt_dir, dataset, f"fold-{fold}"),
                    dataset=dataset, 
                    fold=fold, 
                    reduced=False)
            continue
        for area in config.areas:
            logger.info(f"\n### AREA: {area}\n"+ "-"*(len(area)+10))
            chkpt_dir_dfa = os.path.join(chkpt_dir, dataset, f"fold-{fold}", area)
            os.makedirs(chkpt_dir_dfa, exist_ok=True)
            train_dl_model(chkpt_dir=chkpt_dir_dfa,
                            dataset=dataset,
                            area=area,
                            fold=fold,
                            reduced=False)
        test_linear_probe(chkpt_dir=os.path.join(chkpt_dir, dataset, f"fold-{fold}"),
                            dataset=dataset, 
                            fold=fold, 
                            reduced=False)
        

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
                model = DLModel(latent_dim=config.latent_dim)
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


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--chkpt_dir", required=True, type=str,
    help="Checkpoint dir where all the logs are stored. List of existing checkpoint directories: " \
        + " - ".join(os.listdir(config.path2models)))
    parser.add_argument("-d", "--dataset", required=True, type=str,
                        help="Dataset on which you want to train")
    args = parser.parse_args(argv)
    return args


if __name__ == "__main__":
    
    args = parse_args(sys.argv[1:])
    # train_all_dl_models(chkpt_dir=args.chkpt_dir)
    # test_l2_regularisation()
    # test_linear_probe_cv(chkpt_dir=args.chkpt_dir)

    train_all_dl_models_cv(chkpt_dir=args.chkpt_dir,
                           dataset=args.dataset)
    
    # chkpt_dir = os.path.join(config.path_to_models, "20241206_local_models_pca")
    # for dataset in config.datasets:
    #    test_linear_probe(chkpt_dir=os.path.join(chkpt_dir, dataset),
    #                      dataset=dataset,
    #                      reduced=True,
    #                      fold=None)