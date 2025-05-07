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
from sklearn.preprocessing import StandardScaler
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
    """ Train one MLP from the Champollion embeddings.

    Parameters:
    chkpt_dir: folder where data is saved.
    dataset: the dataset used to train model (asd, bd or scz).
    area: the regional model from which the embeddings are computed.
    reduced: take (or not) the embeddings output by the ACP dimension reduction method.
    fold: the fold on which the model is trained.
    """
        
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

def test_linear_probe(chkpt_dir, dataset, reduced=False, fold=None, gridsearch=False):
    """ Train linear model to make the final prediction from all the local predictions.

    Parameters:
    reduced: take the embeddeings with ACP dimension reduction.
    fold: which fold of the scheme. None is equivalent to the one split scheme.
    gridsearch: select the best hyperparameters for the linear model.
                If False, take the default hyperparameters.
    """
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
    
    # Gridsearch
    if gridsearch:
        cv = PredefinedSplit([-1 for _ in range(len(labels["train"]))] + \
                        [0 for _ in range(len(labels["validation"]))])
        gs = GridSearchCV(LogisticRegression(max_iter=1000, penalty="l2"),
                            param_grid={"C": 10. ** np.arange(-3, 3)},
                            scoring="roc_auc",
                            refit=False,
                            cv=cv, 
                            n_jobs=config.num_workers)
        X = np.concatenate([predictions["train"], predictions["validation"]], axis=0)
        y = np.concatenate([labels["train"], labels["validation"]], axis=0)
        gs.fit(X, y)
        logger.info(f"GridSearch: best score: {gs.best_score_} - best params: {gs.best_params_}")
        best_params = gs.best_params_
    else:
        best_params = {"C": 1.0, "fit_intercept": True}
    # Fit Logistic Regression
    clf = LogisticRegression(max_iter=1000, penalty="l2", **best_params)
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
        logs["fold"].append(fold)
        if gridsearch:
            for param, value in best_params.items():
                logs[param].append(value)
            logs["score"].append(gs.best_score_)
        logs["roc_auc"].append(roc_auc_score(y_score=y_pred[:, 1], y_true=y_true))
        logs["balanced_accuracy"].append(balanced_accuracy_score(y_pred=y_pred.argmax(axis=1), y_true=y_true))

    df_coef = pd.DataFrame({"area": config.areas,
                            "coefficient": clf.coef_.squeeze()})
    df_coef.to_csv(os.path.join(chkpt_dir, f"lrl2_fold-{fold}_epoch-{config.epoch_f}_coef.csv"), 
                   sep=",", index=False)
    df_logs = pd.DataFrame(logs)
    df_logs.to_csv(os.path.join(chkpt_dir,
                                f"lrl2_fold-{fold}_epoch-{config.epoch_f}_test.csv"),
                    sep=",", index=False)

def train_all_dl_models(chkpt_dir, reduced=False):
    """Train local models with one split scheme.
    
    Parameters:
    reduced: take the reduced embeddings as input, i.e., first components of the ACP
    """
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
    """ Train all the local models with the 10-fold cross-validation scheme.
    """
    chkpt_dir = os.path.join(config.path2models, chkpt_dir)
    os.makedirs(chkpt_dir, exist_ok=True)
    setup_logging(level="info", 
                  logfile=os.path.join(chkpt_dir, "train_dl_model.log"))
    # for dataset in config.datasets:
    logger.info(f"\n# DATASET: {dataset}\n" + "-"*(len(dataset)+11))
    for fold in range(config.nb_folds):
        logger.info(f"\n## FOLD: {fold}\n"+ "-"*11)
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
                            reduced=False,
                            gridsearch=True)
        
        

def test_l2_regularisation():
    """ GridSearch to find the best value for weight_decay
    """
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

def train_lrl2_on_all_areas(chkpt_dir):
    """ Train a linear model from the concatenated Champollion embeddings.
    """
    chkpt_dir = os.path.join(config.path2models, chkpt_dir)
    os.makedirs(chkpt_dir, exist_ok=True)
    setup_logging(level="info", 
                  logfile=os.path.join(chkpt_dir, "train_lrl2.log"))
    for dataset in config.datasets:
        logger.info(f"\n# DATASET: {dataset}\n" + "-"*(len(dataset)+11))
        for fold in range(config.nb_folds):
            logger.info(f"\n## FOLD: {fold}\n"+ "-"*11)
            os.makedirs(os.path.join(chkpt_dir, dataset, f"fold-{fold}"), exist_ok=True)
            # Train logistic regressions
            logs = defaultdict(list)
            # Load Data
            logger.info(f"Preprocessing data")
            inputs, labels = load_embeddings_all_areas(dataset, fold)
            scaler = StandardScaler()
            scaler.fit(inputs["train"])
            for split, arr in inputs.items():
                inputs[split] = scaler.transform(arr)
            # Fit Logistic Regression
            logger.info(f"Fitting LogisticRegression")
            cv = PredefinedSplit([-1 for _ in range(len(labels["train"]))] + \
                                 [0 for _ in range(len(labels["validation"]))])
            gs = GridSearchCV(LogisticRegression(max_iter=1000, penalty="l2"),
                                param_grid={"C": 10. ** np.arange(-3, 3)},
                                scoring="roc_auc",
                                cv=cv, 
                                n_jobs=config.num_workers,
                                refit=False)
            gs.fit(np.concatenate([inputs["train"], inputs["validation"]], axis=0),
                   np.concatenate([labels["train"], labels["validation"]], axis=0))
            logger.info(f"Best score: {gs.best_score_} - best params: {gs.best_params_}")
            clf = LogisticRegression(max_iter=10000, penalty="l2", **gs.best_params_)     
            clf.fit(inputs["train"], labels["train"])
            # Test model
            for split in config.splits:
                y_pred = clf.predict_proba(inputs[split])
                y_true = labels[split]
                logs["dataset"].append(dataset)
                logs["fold"].append(fold)
                logs["set"].append(split)
                logs["label"].append("diagnosis")
                logs["reduced"].append(False)
                logs["score"].append(gs.best_score_)
                for param, value in gs.best_params_.items():
                    logs[param].append(value)
                logs["roc_auc"].append(roc_auc_score(y_score=y_pred[:, 1], y_true=y_true))
                logs["balanced_accuracy"].append(balanced_accuracy_score(y_pred=y_pred.argmax(axis=1), y_true=y_true))

            df_coef = pd.DataFrame(data=clf.coef_.reshape(len(config.areas), 256), 
                                   columns=[f"coef_latent_dim_{i}" for i in range(256)])
            df_coef.insert(0, "area", config.areas)
            df_coef.to_csv(os.path.join(chkpt_dir, dataset, f"fold-{fold}",
                                 f"lrl2_fold-{fold}_coef_.npy"),
                                 sep=",", index=False)
            df_logs = pd.DataFrame(logs)
            df_logs.to_csv(os.path.join(chkpt_dir, dataset, f"fold-{fold}",
                                        f"lrl2_fold-{fold}_test.csv"),
                            sep=",", index=False)

def load_embeddings_all_areas(dataset, fold):
    """ Load all the Champollion embeddings to train a linear model.
    """
    studies = {
    "asd": ["abide1", "abide2"],
    "bd": ["biobd", "bsnip1", "cnp", "candi"],
    "scz": ["schizconnect-vip-prague", "bsnip1", "cnp", "candi"]
    }[dataset]
    unique_keys = {"asd": ["participant_id", "session", "run", "study"],
                "bd": ["participant_id", "session", "study"],
                "scz": ["participant_id", "session", "study"]}[dataset]
    train_val_scheme = f"{dataset}_age_sex_diagnosis_site_stratified_10-fold.csv"
    target_mapping = {"control": 0,     
                "asd": 1,
                "bd": 1, "bipolar disorder": 1, "psychotic bd": 1, 
                "scz": 1}
    embeddings, metadata = None, None
    for area in config.areas:
        df = pd.concat([pd.read_csv(os.path.join(config.path2data, area,
                                                    f"{s}_skeleton_{area.lower()}_participants.csv"),
                                                        dtype=config.id_types) 
                                        for s in studies],
                            ignore_index=True, sort=False)
        if metadata is None:
            metadata = df
        else: # sanity check
            assert (df[unique_keys] == metadata[unique_keys]).all().all()


        arr = np.vstack([np.load(os.path.join(config.path2data, area,
                                            f"{s}_skeleton_{area.lower()}.npy"))
                        for s in studies])
        if embeddings is None:
            embeddings = arr
        else:
            embeddings = np.hstack((embeddings, arr))
    scheme = pd.read_csv(os.path.join(config.path2schemes, train_val_scheme), dtype=config.id_types)
    metadata = metadata.merge(scheme, how="left", on=unique_keys, validate="1:1")

    inputs, labels = {}, {}
    for split in config.splits:
        inputs[split] = embeddings[metadata[f"fold-{fold}"] == split]
        labels[split] = metadata.loc[metadata[f"fold-{fold}"] == split, "diagnosis"].replace(target_mapping).values 
    return inputs, labels


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
    train_all_dl_models_cv(chkpt_dir=args.chkpt_dir,
                           dataset=args.dataset)
    # train_all_dl_models(chkpt_dir=args.chkpt_dir)
    # test_l2_regularisation()
    # test_linear_probe_gridsearch(chkpt_dir=args.chkpt_dir)
    # train_lrl2_on_all_areas(chkpt_dir=args.chkpt_dir)
    # chkpt_dir = os.path.join(config.path_to_models, "20241206_local_models_pca")
    # for dataset in config.datasets:
    #    test_linear_probe(chkpt_dir=os.path.join(chkpt_dir, dataset),
    #                      dataset=dataset,
    #                      reduced=True,
    #                      fold=None)