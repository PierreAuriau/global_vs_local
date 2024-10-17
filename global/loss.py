#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func


class BarlowTwinsLoss(nn.Module):
    """ Implementation from : 
        J. Zbontar, et al., Barlow Twins: Self-Supervised Learning via Redundancy Reduction, ICML 2021
        Available from : https://proceedings.mlr.press/v139/zbontar21a.html.
    """

    def __init__(self, correlation='cross', lambda_param=5e-3):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        self.correlation = correlation

    def forward(self, z_a, z_b):
        # normalize repr. along the batch dimension
        # beware: normalization is not robust to batch of size 1
        # if it happens, it will return a nan loss
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0) # NxD

        N = z_a.size(0)
        D = z_a.size(1)

        if self.correlation=='cross':
            # cross-correlation matrix
            c = torch.mm(z_a_norm.T, z_b_norm) / N # DxD
            # loss
            c_diff = (c - torch.eye(D, device=c.device)).pow(2) # DxD
            # multiply off-diagonal elems of c_diff by lambda
            c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param
            loss = c_diff.sum()
        elif self.correlation=='auto':
            # auto-correlation matrix
            c1 = torch.mm(z_a_norm.T, z_a_norm) / N # DxD
            c2 = torch.mm(z_b_norm.T, z_b_norm) / N # DxD
            c = (c1.pow(2) + c2.pow(2)) / 2
            c[torch.eye(D, dtype=bool)]=0
            redundancy_loss = c.sum()
            # cross-correlation matrix
            c = torch.mm(z_a_norm.T, z_b_norm) / N # DxD
            # loss
            c_diff = (c - torch.eye(D,device=self.device)).pow(2) # DxD
            c_diff[~torch.eye(D, dtype=bool)]=0
            loss = c_diff.sum()
            loss += self.lambda_param*redundancy_loss
        else:
            raise ValueError("Wrong correlation specified in BarlowTwins\
                             config: use cross or auto.")

        return loss
