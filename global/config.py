#!/usr/bin/env python

import os

class Config:
  def __init__(self):

    self.path_to_project = "/neurospin/psy_sbox/analyses/2024_pauriau_global_vs_local"
    self.path_to_schemes = os.path.join(self.path_to_project, "data", "schemes")
    self.path_to_models = os.path.join(self.path_to_project, "models", "global")
    # self.path_to_data = os.path.join(self.path_to_project, "data", "global")
    self.path_to_masks = os.path.join(self.path_to_project, "data", "masks")
    # FIXME : path to dataset
    self.path2data = "/neurospin/psy_sbox/analyses/202205_predict_neurodev/data/skeletons/ukbiobank/arrays/without_channel"
    self.path2schemes = "/neurospin/psy_sbox/analyses/2024_pauriau_global_vs_local/data/schemes"

    self.path2clinicaldata = "/neurospin/psy_sbox/analyses/2024_pauriau_global_vs_local/data/global"
    
    self.path2models = "/neurospin/psy_sbox/analyses/2024_pauriau_global_vs_local/models/global"

    # self.path_to_data = "/home_local/pa267054"
    self.path_to_data = "/home_local/pa267054/data_global"

    self.datasets = ("asd", "bd", "scz")
    self.splits = ("train", "validation", "internal_test", "external_test")
    
    self.n_embedding = 256
    self.correlation_bt = 'cross'
    self.lambda_bt = 1.0
    self.lr = 1e-4
    self.nb_epochs = 300
    self.data_augmentation = "cutout"

    self.lr_ft = 1e-4
    self.nb_epochs_ft = 100
    self.weight_decay_ft = 5e-3

    self.batch_size = 32
    self.num_workers = 8

    # ID types in participants dataframe
    self.id_types = {"participant_id": str,
                     "session": int,
                     "run": int,
                     "acq": int}
    
    # XAI
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
