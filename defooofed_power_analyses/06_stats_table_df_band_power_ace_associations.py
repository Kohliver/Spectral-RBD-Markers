#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 13:44:41 2025

@author: okohl

Get tabesls with stats for defooofed spectrax ACE correlations/

"""

import pickle
import numpy as np
import pandas as pd


# Set Dirs
datdir = '/ohba/pi/knobre/okohl/RBD_x_PD/Glasser52/rest/data/static/df_band_power/associations/ace'
outdir = '/ohba/pi/knobre/okohl/RBD_x_PD/Glasser52/rest/results/static/metrics/df_band_power/associations/ace/stats'
dvs = ["theta","alpha","beta"]
contrasts = ["age","education",'HC','PD','RBD','All']

# Load Labels
labels = pickle.load(open("/ohba/pi/knobre/okohl/Glasser52/Labels_short.p","rb"))
label_nrs = np.arange(1,53)

for i, cont in enumerate(contrasts):
    ts = np.load(f"{datdir}/contrast_{i}.npy")
    pvalues = np.load(f"{datdir}/contrast_{i}_pvalues.npy")
    
    for ind, dv in enumerate(dvs):
    
        # Only extract significant parcel values
        mask = pvalues[ind] < .05
        t_sig = np.round(ts[ind,mask],2)
        p_sig = np.round(pvalues[ind,mask],3)
        label_nrs_sig = label_nrs[mask]
        
        labels_sig = [lab for lab, m in zip(labels, mask) if m]
        
        df = pd.DataFrame({
            "Parcel Nr.": label_nrs_sig,
            "Parcel": labels_sig,
            "T-Statistic": t_sig,
            "p-Value": p_sig
            })
              
        df.to_csv(f"{outdir}/cont-{cont}_dv-{dv}_statistics.csv", index=False )