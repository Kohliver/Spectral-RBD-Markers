#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:49:18 2024

@author: okohl

Script Assessing Fooof outputs.
Outlier detection is 
"""

import numpy as np
from scipy.io import loadmat
from scipy.stats import ttest_ind, zscore
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
import seaborn as sns

import sys
sys.path.append(".../RBD_x_PD/Glasser52/rest/scripts/helpers/")
sys.path.append(".../RBD_x_PD/Glasser52/rest/scripts/dynamic/all/helper")
from parameters import ROIS_SHORT
from plotting import prepare_plotting
import old_power


def find_outliers_iqr(values, factor=1.5):
    """
    Finds outlier indices using the IQR method.
    factor: multiplier for IQR to set cutoff range
    """
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    return np.where((values < lower_bound) | (values > upper_bound))[0]


# Set dirs
indir = '.../RBD_x_PD/Glasser52/rest/data/static/psd'
metric_dir = '.../RBD_x_PD/Glasser52/rest/data/static/metrics'
plot_dir = '.../RBD_x_PD/Glasser52/rest/results/static/spectra/fooof_check/overview' 

# get group list
demo = pd.read_csv('.../RBD_x_PD/Glasser52/rest/data/demographics/demographics_rest.csv')
group_list = demo['Group'].values

# --- Load Parcel Coordinates ---
coord_dir = '.../Glasser52/Parcellation_Info/';
ROIcenters = loadmat(coord_dir + 'Glasser52_8mm_centers_dist_side_info.mat')['c']
ROIy = ROIcenters[:,1]

# Get y-Ranks for ROIs
tmp = (-ROIy).argsort() # Sorted indices // from front to back
ROIy_rank = np.empty_like(tmp)
ROIy_rank[tmp] = np.arange(len(ROIy))
z_yrank = zscore(ROIy_rank)

# Load Interpolated Spectra and f
f = np.load(f"{indir}/f.npy")
psd = np.load(f"{indir}/interp_psd.npy")
f_mask = np.logical_and(f>=2,f<=40)
f =  f[f_mask]
psd = psd[:,:,f_mask]

defooofed_f = np.load(f"{indir}/defooofed_f.npy")
defooofed_psd = np.load(f"{indir}/defooofed_psd.npy")
aperiodic_exponent = np.load(f"{metric_dir}/aperiodic_exponent.npy")
aperiodic_psd = np.load(f"{indir}/aperiodic_psd.npy")
rsq = np.load(f"{indir}/fooof_rsquared.npy")
err = np.load(f"{indir}/fooof_error.npy")
offsets = np.load(f"{indir}/aperiodic_offsets.npy")

# Prepar Plotting
group_labels, group_colors, _ , cropped_cmaps = prepare_plotting(inverse_cmp=False)
cmaps = [cropped_cmaps[0], cropped_cmaps[1], cropped_cmaps[0], cropped_cmaps[3], cropped_cmaps[4]]

# Run outlier detection
metrics = [err.mean(axis=1), rsq.mean(axis=1)]
outliers = [find_outliers_iqr(metric, factor=3) for metric in metrics]

# Mark outlier participant with poor fits
fooof_outl = np.zeros(len(group_list))
fooof_outl[36] = 1
fooof_outl[84] = 1
fooof_outl = fooof_outl > 0

# Get Parameters for outlier
outlier_list = group_list[fooof_outl]
out_err = err[fooof_outl]
out_rsq = rsq[fooof_outl]
out_offsets = offsets[fooof_outl]
out_aperiodic_exponent = aperiodic_exponent[fooof_outl]

# Mask for participant with bad Fooof Fits
group_list = group_list[~fooof_outl]
err = err[~fooof_outl]
rsq = rsq[~fooof_outl]
offsets = offsets[~fooof_outl]
aperiodic_exponent = aperiodic_exponent[~fooof_outl]

    
#%% Average across participants
out_metrics = [out_err.mean(axis=1), out_rsq.mean(axis=1), out_offsets.mean(axis=1), out_aperiodic_exponent.mean(axis=1)]
metrics = [err.mean(axis=1), rsq.mean(axis=1), offsets.mean(axis=1), aperiodic_exponent.mean(axis=1)]
metric_labels = ['Error','R-Squared','Offset','Aperiodic Exponent']

for metric, out_metric, metric_label in zip(metrics, out_metrics, metric_labels):

    # Ttests comparing grou[s]
    HC1 = metric[group_list == 1].flatten()
    PD = metric[group_list == 2].flatten()
    HC2 = metric[group_list == 3].flatten()
    RBD = metric[group_list == 4].flatten()
    
    t_pd, p_pd = ttest_ind(HC1,PD)
    t_rbd, p_rbd = ttest_ind(HC2,RBD)
    
    ts = [t_pd,t_rbd]
    ps = [p_pd,p_rbd]
    
    print(ts)
    print(ps)
    
    # Bring Data into long format
    group_long = group_list
    metric_long = metric.flatten('F')
    #state_long = np.repeat(ROIy_rank)
    
    df = pd.DataFrame(np.vstack([group_long,metric_long]).T,columns=['group','metric'])
    df_out = pd.DataFrame(np.vstack([outlier_list,out_metric]).T,columns=['group','metric'])
    
    # --- Make Split Bar Plot for all States - w Stripe Plots ---
    
    # Start Plotting
    fig = plt.figure(dpi=600, figsize=(7.4,4.8))
    ax = fig.add_subplot()
                   
    # make plots
    sns.boxplot(x="group", y='metric', data=df, palette=group_colors, width=.5, ax = ax, showfliers=False)
    for i in range(4):
        sns.stripplot(x="group", y='metric',data=df[df['group']==i+1], 
                      color='#3b3b3b', size=5, alpha=1, dodge=False, 
                      legend=False,ax = ax)
        if i == 0:
            sns.stripplot(x="group", y='metric', data=df_out[df_out['group']==i+1], 
                          color="#A62700", size=8, dodge=False, 
                          legend=False,ax = ax)
        if i == 2:
            sns.stripplot(x="group", y='metric', data=df_out[df_out['group']==i+1], 
                          color="#E96245", size=8, dodge=False, 
                          legend=False,ax = ax)
    
    # Labels and Ticks
    ax.set_xlabel('', fontsize=16)
    ax.set_xticklabels(group_labels, fontsize=16) #becasue of np.arrang does not include end of specified range
    ax.set_ylabel(metric_label, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    plt.locator_params(axis='y',nbins=5)
            
    # Remove Box Around Subplot
    sns.despine(ax=ax, top=True, right=True, left=False,
                bottom=False, offset=True, trim=False)
    
    # # Save Plot    
    plt.savefig(f'{plot_dir}/{metric_label}_parcel_overview_wOutliers.svg',
                bbox_inches = 'tight')
    

#%% PLot fooof fit per parcel and participant
# Participant 36 shows a lot of bad fits 

bad_fits = np.where(np.min(rsq,axis=1) < .9)[0]
bad_sub = np.where(np.sum(rsq < .9,axis=1) > 3)
bad_parc = np.where((np.sum(rsq < .9,axis=0) > 2))[0]

bad_sub = np.where(rsq.mean(axis=1) < .975)[0]
bad_parc = np.where(rsq.mean(axis=0) < .975)[0]

sns.stripplot(rsq.mean(axis=0))

for iParc in bad_parc:
    print(f'{ROIS_SHORT[iParc]} is bad.')

nSub = psd.shape[0]
nParc = psd.shape[1]

for iSub in bad_fits: # loop across sub
    print(f'Running Sub{iSub}')

    for iParc in range(nParc):
        line1 = np.log10(psd[iSub,iParc])
        line2 = aperiodic_psd[iSub,iParc]
        line3 = aperiodic_psd[iSub,iParc] + defooofed_psd[iSub,iParc]
        
        # Start Plotting
        fig = plt.figure()
        ax = fig.add_subplot()
        
        ax.plot(f,line1)
        ax.plot(f,line2, color='blue',linestyle='--',alpha=.7)
        ax.plot(f,line3, color = 'red',alpha=.7)
        
        ax.text(x=25, y=-1.5, s=f'Error: {np.round(err[iSub,iParc],3)}', zorder=10, size=14 )
        ax.text(x=25, y=-1.6, s=f'R-Squared: {np.round(rsq[iSub,iParc],2)}', zorder=10, size=14 )
        ax.text(x=25, y=-1.7, s=f'Parcel: {ROIS_SHORT[iParc]}', zorder=10, size=14 )
