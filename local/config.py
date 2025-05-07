#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

class Config:
  def __init__(self):
    # Paths to directories
    self.path_to_project = "/neurospin/psy_sbox/analyses/2024_pauriau_global_vs_local"
    self.path_to_embeddings = "/neurospin/dico/data/deep_folding/current/models/Champollion_V0"
    self.path_to_data = os.path.join(self.path_to_project, "data", "local")
    self.path_to_models = os.path.join(self.path_to_project, "models", "local")
    self.path_to_schemes = os.path.join(self.path_to_project, "data", "schemes")
    
    # old version
    self.path2data = os.path.join(self.path_to_project, "data", "local")
    self.path2schemes = os.path.join(self.path_to_project, "data", "schemes")
    self.path2models = os.path.join(self.path_to_project, "models", "local")

    # Latent dim of the local encoder Champollion VO
    self.latent_dim = 256
    self.n_components = 20 # for the ACP dimension reduction
    
    # Parameters for MLPs
    self.lr = 1e-4
    self.weight_decay = 5e-1
    self.nb_epochs = 100
    self.epoch_f = self.nb_epochs - 1
    self.batch_size = 64
    self.num_workers = 8
    self.nb_folds = 10

    # all dataset
    self.datasets = ("asd", "bd", "scz")

    # all splits
    self.splits = ("train", "validation", "internal_test", "external_test")

    # ID types in participants dataframe
    self.id_types = {"participant_id": str,
                     "session": int,
                     "run": int,
                     "acq": int}

    # all the studies
    self.studies = ("abide1", 
                    "abide2", 
                    "biobd", 
                    "bsnip1", 
                    "candi", 
                    "cnp", 
                    "schizconnect-vip-prague")
    
    # all the local areas
    self.areas = ('SC-sylv_left',
                  'SFinter-SFsup_left',
                  'STi-STs-STpol_right',
                  'FColl-SRh_right',
                  'SC-sylv_right',
                  'Lobule_parietal_sup_left',
                  'SFint-SR_left',
                  'SPoC_left',
                  'SFinf-BROCA-SPeCinf_right',
                  'SsP-SPaint_left',
                  'SOr-SOlf_left',
                  'SFinf-BROCA-SPeCinf_left',
                  'SFmedian-SFpoltr-SFsup_right',
                  'FColl-SRh_left',
                  'STi-STs-STpol_left',
                  'SFint-FCMant_left',
                  'FCLp-subsc-FCLa-INSULA_right',
                  'OCCIPITAL_right',
                  'SC-SPeC_right',
                  'SOr-SOlf_right',
                  'CINGULATE_left',
                  'SFint-SR_right',
                  'CINGULATE_right',
                  # 'FCLp-SGSM_left',
                  # 'FCLp-SGSM_right',
                  # 'fronto-parietal_medial_face_left',
                  # 'fronto-parietal_medial_face_right',
                  # 'STs-SGSM_left',
                  # 'STs-SGSM_right',
                  'ScCal-SLi_left',
                  'FPO-SCu-ScCal_right',
                  'FPO-SCu-ScCal_left',
                  'OCCIPITAL_left',
                  'SC-SPeC_left',
                  'STs_right',
                  'FIP_left',
                  'STsbr_right',
                  'SC-SPoC_right',
                  'SPeC_left',
                  'SPoC_right',
                  'STi-SOTlat_right',
                  'FCMpost-SpC_left',
                  'SFint-FCMant_right',
                  'SFmedian-SFpoltr-SFsup_left',
                  'SPeC_right',
                  'SFmarginal-SFinfant_left',
                  'FIP_right',
                  'FCLp-subsc-FCLa-INSULA_left',
                  'STsbr_left',
                  'SsP-SPaint_right',
                  'ScCal-SLi_right',
                  'Lobule_parietal_sup_right',
                  'ORBITAL_left',
                  'SFmarginal-SFinfant_right',
                  'STi-SOTlat_left',
                  'SFinter-SFsup_right',
                  'STs_left',
                  'FCMpost-SpC_right',
                  'ORBITAL_right',
                  'SC-SPoC_left')
