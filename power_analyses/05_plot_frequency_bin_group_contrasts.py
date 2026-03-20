#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 09:19:08 2024

@author: okohl

Plot GLMs calculating group contrasts for power at each freuqency bin.
T-statistics per frequency band a represented in Figure 4.
"""

import glmtools as glm
import numpy as np
import pandas as pd
from scipy.io import loadmat,savemat
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(".../RBD_x_PD/Glasser52/rest/scripts/helpers/")
from plotting import prepare_plotting
import old_power


# Set Dirs
datdir = '.../RBD_x_PD/Glasser52/rest/data/static/spect/contrasts'
outdir = '...RBD_x_PD/Glasser52/rest/results/static/metrics/spect/glms'
metric_dir = '.../RBD_x_PD/Glasser52/rest/data/static/psd'

# --- Set TStat Parameters ---
tstat_args = {'hat_factor': 5e-3, 'varcope_smoothing': 'medfilt',
              'window_size': 15, 'smooth_dims': 1}

group_labels, group_colors, _ , cropped_cmaps = prepare_plotting(inverse_cmp=True)
cmaps = [cropped_cmaps[1], cropped_cmaps[3],cropped_cmaps[0]]



# --- Load Parcel Coordinates ---
coord_dir = '/ohba/pi/knobre/okohl/forChet/Parcellation/Glasser52/Parcellation_Info/';
ROIcenters = loadmat(coord_dir + 'Glasser52_8mm_centers_dist_side_info.mat')['c']
ROIy = ROIcenters[:,1]

# Get y-Ranks for ROIs
tmp = (-ROIy).argsort() # Sorted indices // from front to back
ROIy_rank = np.empty_like(tmp)
ROIy_rank[tmp] = np.arange(len(ROIy))

# --- Load Freuqency bins ---
f = np.load(f"{metric_dir}/f.npy")
f = f[f<=35]

# --- Grab palettes ---
parc = []
parc.append(plt.cm.Greys(np.linspace(.2,1,52)))
parc.append(plt.cm.Blues(np.linspace(.2,1,52)))
parc.append(plt.cm.Reds(np.linspace(.2,1,52)))
parc.append(plt.cm.Oranges(np.linspace(.2,1,52)))

for i, name in enumerate(["age","PD-HC",'RBD-HC','PD-RBD']):
    ts = np.load(f"{datdir}/contrast_{i}.npy")
    pvalues = np.load(f"{datdir}/contrast_{i}_pvalues.npy")
    thresh = np.load(f"{datdir}/contrast_{i}_thresh.npy")
    Parcel_palette = parc[i]

    # ----- GLM PLOT ----- 
    ax = np.zeros(8, dtype=object)
    fig, ax[1] = plt.subplots(dpi=300)
    
    for iRoi in range(ts.shape[0]):
        ax[1].plot(f, ts[iRoi,:].T,linewidth = .7, color = Parcel_palette[ROIy_rank[iRoi]])
     

    for ind, label in enumerate(['p=0.05']):    
        ax[1].plot(f,  np.ones_like(f)*thresh[ind],linestyle='--',color="#636363", linewidth=.8, label=label,zorder=1)
        ax[1].plot(f,  np.ones_like(f)*-thresh[ind],linestyle='--',color="#636363",linewidth=.8,zorder=1)
    
    # add labels to plots
   # ax[1].legend(bbox_to_anchor=(1, 1),fontsize = 12, frameon=False)
    ax[1].set_xlim([1, 35])
    ax[1].set_ylim([-6.7,5.5])
    ax[1].set_xticks([5,10,15,20,25,30])
    ax[1].set_yticks([-5,-2.5,0,2.5,5])
        
    #ax[1].set_title(name,fontsize = 16)
    ax[1].set_ylabel('T-Statistic', fontsize = 16)
    ax[1].set_xlabel('Frequency (Hz)', fontsize = 16)
    ax[1].tick_params(axis='both', which='major', labelsize=12)    
    
    # Add lines indicating freqs to plot
    ax[1].axhline(0, color = 'grey', linestyle = '--', linewidth = .5, zorder = 1)
    ax[1].axvline(5, color = 'grey', linestyle = '--', linewidth = .5, zorder = 1)
    ax[1].axvline(8, color = 'grey', linestyle = '--', linewidth = .5, zorder = 1)
    ax[1].axvline(13, color = 'grey', linestyle = '--', linewidth = .5, zorder = 1)
    ax[1].axvline(30, color = 'grey', linestyle = '--', linewidth = .5, zorder = 1)
    
    # Remove Box Around Subplot
    sns.despine(ax=ax[1], top=True, right=True, left=False, bottom=False)

    plt.savefig(f'{outdir}/{name}_spect_glm_overview.svg',transparent = False, format='svg')


   
