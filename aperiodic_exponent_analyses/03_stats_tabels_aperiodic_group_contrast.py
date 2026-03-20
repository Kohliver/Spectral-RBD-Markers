#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 13:20:22 2025

@author: okohl

Tabels with statsistic for group contrastst of aperiodic component.
Presented in SI of manuscript.

"""

import pickle
import numpy as np
import pandas as pd

# Set Dirs
datdir = '.../RBD_x_PD/Glasser52/rest/data/static/aperiodic/contrasts'
outdir = '.../RBD_x_PD/Glasser52/rest/results/static/metrics/aperiodic/glms/stats'
contrasts = ["age","PD-HC",'RBD-HC','PD-RBD']

# Load Labels
labels = pickle.load(open(".../Glasser52/Labels_short.p","rb"))
label_nrs = np.arange(1,53)

for i, cont in enumerate(contrasts):
    ts = np.load(f"{datdir}/contrast_{i}.npy")
    pvalues = np.load(f"{datdir}/contrast_{i}_pvalues.npy")
    
    # Only extract significant parcel values
    mask = pvalues < .05
    t_sig = np.round(ts[mask],2)
    p_sig = np.round(pvalues[mask],3)
    label_nrs_sig = label_nrs[mask]
    
    labels_sig = [lab for lab, m in zip(labels, mask) if m]
    
    df = pd.DataFrame({
        "Parcel Nr.": label_nrs_sig,
        "Parcel": labels_sig,
        "T-Statistic": t_sig,
        "p-Value": p_sig
        })
          
    df.to_csv(f"{outdir}/cont-{cont}_dv-aperiodic_statistics.csv", index=False )
