#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 12:52:23 2023

@author: okohl

Get Metrics descring spectral features.
Delta/Theta/Alpha/Beta Power of PSD

"""

from osl_dynamics.analysis import power
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append(".../RBD_x_PD/Glasser52/scripts/helpers/")
from plotting import data_to_longformat, prepare_plotting, crop_psd, psd_plot

# Set dirs
indir = '.../RBD_x_PD/Glasser52/rest/data/static/psd'
metric_dir = '.../RBD_x_PD/Glasser52/rest/data/static/metrics'

# Get Groups
demo = pd.read_csv('.../RBD_x_PD/Glasser52/rest/data/demographics/demographics_rest.csv')
group = demo['Group'].values

# Group Vars
_, group_colors, _,_ = prepare_plotting()

#%% Normal Spectra

# Load Interpolated Spectra and f
f = np.load(f"{indir}/f.npy")
psd = np.load(f"{indir}/interp_psd.npy")

# Band Power
power_ = []
for band in [[2, 4], [5, 8], [8, 12], [13, 30]]:
    power_.append(power.variance_from_spectra(f, psd, frequency_range=band))
power_ = np.swapaxes(power_, 0, 1)

# Save
np.save(f"{metric_dir}/band_power.npy", power_)


#%% Defooofed

# Load defooofed data
f = np.load(f"{indir}/defooofed_f.npy")
psd = np.load(f"{indir}/defooofed_psd.npy")

# --- Band Power defooofed ---
def_power_ = []
for band in [[2, 4], [5, 7], [8, 12], [13, 30]]:
    def_power_.append(power.variance_from_spectra(f, psd, frequency_range=band))
def_power_ = np.swapaxes(def_power_, 0, 1)

# Save
np.save(f"{metric_dir}/defooofed_band_power.npy", def_power_)

