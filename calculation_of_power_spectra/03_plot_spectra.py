#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 12:51:21 2023

@author: okohl

Plot Spectra.
Obtained figures are not presented in yhe manuscript and were only used for quality control.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append(".../RBD_x_PD/Glasser52/rest/scripts/helpers/")
from plotting import data_to_longformat, prepare_plotting, crop_psd, psd_plot


# Set dirs
indir = '.../RBD_x_PD/Glasser52/rest/data/static/psd'
metric_dir = '.../RBD_x_PD/Glasser52/rest/data/static/metrics'

#%% Grand Average

# Load Interpolated Spectra and f
f = np.load(f"{indir}/f.npy")
psd = np.load(f"{indir}/interp_psd.npy")
psd = psd.mean(axis=1) # mean across parcels

# get group list
demo = pd.read_csv('/ohba/pi/knobre/okohl/RBD_x_PD/Glasser52/rest/data/demographics/demographics_rest.csv')
group_list = demo['Group'].values

# Prepare Data for plotting
f, mean_psd = crop_psd(psd,f,f_range=[2,45])


# --- All Groups Comparison ---
plot_in = data_to_longformat(mean_psd,f,group_list)

# Group Vars
group_labels, group_colors, group_cmaps,_ = prepare_plotting()

# Plotting
fig, ax = plt.subplots(dpi=300)
psd_plot(plot_in[:],group_colors, group_labels,ax)
plt.savefig('.../RBD_x_PD/Glasser52/rest/results/static/spectra/avg_all.png')


# --- HC 1 vs HC2 ---
mask = np.logical_or(group_list == 1,group_list==3)
plot_in = data_to_longformat(mean_psd[mask],f,group_list[mask])

# Group Vars
group_labels = ['HC1','HC2']
group_colors_in = ['Black','Grey']

# Plotting
fig, ax = plt.subplots(dpi=300)
psd_plot(plot_in,group_colors_in, group_labels,ax)
plt.savefig('/ohba/pi/knobre/okohl/RBD_x_PD/Glasser52/rest/results/static/spectra/avg_HC1_x_HC2.png')


# --- HC 1 vs PD ---
mask = np.logical_or(group_list == 1,group_list==2)
plot_in = data_to_longformat(mean_psd[mask],f,group_list[mask])

# Group Vars
group_labels = ['HC','PD']
group_colors_in = [group_colors[0],group_colors[1]]

# Plotting
fig, ax = plt.subplots(dpi=300)
psd_plot(plot_in,group_colors_in, group_labels,ax)
plt.savefig('.../RBD_x_PD/Glasser52/rest/results/static/spectra/avg_HC_x_PD.png')


#  --- HC2 vs RBD ---
mask = np.logical_or(group_list == 3,group_list==4)
plot_in = data_to_longformat(mean_psd[mask],f,group_list[mask])

# Group Vars
group_labels = ['HC','RBD']
group_colors_in = [group_colors[2],group_colors[3]]

# Plotting
fig, ax = plt.subplots(dpi=300)
psd_plot(plot_in,group_colors_in, group_labels,ax)
plt.savefig('.../RBD_x_PD/Glasser52/rest/results/static/spectra/avg_HC_x_RBD.png')



