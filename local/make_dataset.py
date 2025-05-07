# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create dataset from Champollion embeddings with:
* a numpy array of size (N, 256) for the embeddings (N: nb of subjects)
* a pandas dataframe with the metadata of each subject (in particular the participant_id and the diagnosis)
The array and the dataframe have the same order.
"""
import os
import pandas as pd
import numpy as np
import glob
from multiprocessing import Pool
import itertools

from config import Config

config = Config()

def make_dataset(area, study):
    print(f"area: {area} | study: {study}")
    # get model
    model = [m for m in os.listdir(os.path.join(config.path_to_embeddings, area))
            if os.path.isdir(os.path.join(config.path_to_embeddings, area, m))][0]
    list_embeddings = glob.glob(os.path.join(config.path_to_embeddings, area, model,
                                             "*_random_embeddings"))
    if len(list_embeddings) == 0:
        model += "/" + [m for m in os.listdir(os.path.join(config.path_to_embeddings, area, model))
                        if os.path.isdir(os.path.join(config.path_to_embeddings, area, model, m))][0]
        list_embeddings = glob.glob(os.path.join(config.path_to_embeddings, area, model,
                                                "*_random_embeddings"))
    # read dataframe
    df = pd.read_csv(os.path.join(config.path_to_project,
                                  "data", 
                                  "global",
                                  f"{study}_t1mri_skeleton_participants.csv"),
                     sep=",", dtype=config.id_types)
    
    embeddings = pd.read_csv(os.path.join(config.path_to_embeddings,
                                            area,
                                            model,
                                            f"{study}_random_embeddings",
                                            "full_embeddings.csv"))
    print("n_sbj with embeddings:", len(embeddings))
    print("n_sbj in study:", len(df))

    # add keys to embedding dataframe
    embeddings["model"] = model
    embeddings["participant_id"] = embeddings["ID"].str.extract("sub-([a-zA-Z0-9]+)")[0]
    embeddings["session"] = embeddings["ID"].str.extract("ses-(?:[Vv])?([0-9]+)")[0]
    embeddings["acq"] = embeddings["ID"].str.extract("acq-([0-9]+)")[0]
    embeddings["run"] = embeddings["ID"].str.extract("run-([0-9]+)")[0]

    keys = ["participant_id"]
    for k in ("session", "acq", "run"):
        if embeddings[k].notnull().any():
            keys.append(k)
        else:
            embeddings = embeddings.drop(columns=[k])
    
    # reorder embeddings according to df
    if len(keys) == 1:
        key = keys[0]
        new_index = df[key]
        isin = new_index.isin(embeddings[key])
        if isin.sum() != len(new_index):
            print(f"Nb of sbj without embeddings: {(~isin).sum()}")
            df[~isin].to_csv(os.path.join(config.path2data,
                                          f"{study}_{area.lower()}_participants_without_embeddings.csv"),
                                sep=",", index=False)
            df = df[isin]
            new_index = df[key]            
        embeddings = embeddings.set_index(key)
        embeddings = embeddings.reindex(new_index)
        embeddings = embeddings.reset_index()
    else:
        new_index = df[keys].apply(lambda row: "_".join(row.values.astype(str)), axis=1)
        embeddings["keys"] = embeddings[keys].apply(lambda row: "_".join(row.values.astype(str)), axis=1)
        isin = new_index.isin(embeddings["keys"])
        if isin.sum() != len(new_index):
            print(f"Nb of sbj without embeddings: {(~isin).sum()}")
            df[~isin].to_csv(os.path.join(config.path2data,
                                          f"{study}_{area.lower()}_participants_without_embeddings.csv"),
                                sep=",", index=False)
            df = df[isin]
            new_index = df[keys].apply(lambda row: "_".join(row.values.astype(str)), axis=1)
        embeddings = embeddings.set_index("keys")
        embeddings = embeddings.reindex(new_index)
        embeddings = embeddings.reset_index(drop=True)
    
    # create numpy array from dataframe
    arr = embeddings[[f"dim{i+1}" for i in range(config.latent_dim)]].values
    assert not np.isnan(arr).any(), "Found NaN in the array"    
    assert np.all(arr.shape == (len(df), config.latent_dim)), f"Wrong array shape: {arr.shape}"

    # saving
    os.makedirs(os.path.join(config.path2data, area), exist_ok=True)
    df = df.drop(columns="ni_path")
    df["area"] = area
    df["model"] = model
    df.to_csv(os.path.join(config.path2data, area, 
                            f"{study}_skeleton_{area.lower()}_participants.csv"),
                sep=",", index=False)
    np.save(os.path.join(config.path2data, area,   
                            f"{study}_skeleton_{area.lower()}.npy"),
            arr.astype(np.float32))
    print("Saving into:", os.path.join(config.path2data, area))
    print("\t*", f"{study}_skeleton_{area.lower()}_participants.csv")
    print("\t*",f"{study}_skeleton_{area.lower()}.npy" )

if __name__ == "__main__":
    serial = False
    parallel = True
    # Serial
    if serial:
        for area in config.areas:
            for study in config.studies:
                make_dataset(area=area, study=study)
    # Parallel
    if parallel:
        with Pool() as pool:
            pool.starmap(make_dataset, 
                         [(a, s) for a,s in itertools.product(config.areas, 
                                                              config.studies)])
