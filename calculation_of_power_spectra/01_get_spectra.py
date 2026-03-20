#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 23:30:24 2023

@author: okohl

Calculate Spectra with Welchs Method.
Interpolate across 17Hz bin with fooofs interpolation.
Notch filter was applied at 17Hz becaus of fan artefact in RBD dataset.
"""

import numpy as np
import matplotlib.pyplot as plt

from osl_dynamics.data import Data
from osl_dynamics.analysis import static

from fooof.utils import interpolate_spectra

# define dirs
indir = '.../RBD_x_PD/Glasser52/rest/data/inputs'
outdir = '.../RBD_x_PD/Glasser52/rest/data/static/psd'

# load Data
data = Data([f"{indir}/array{i:03d}.npy" for i in range(0, 115)], n_jobs=4)

# calculate psd
f, psd = static.welch_spectra(
    data=data.time_series(),
    window_length=500,
    sampling_frequency=250,
    frequency_range=[1,45],
    n_jobs=4,
)

np.save(f"{outdir}/f.npy", f)
np.save(f"{outdir}/psd.npy", psd)

# Interpolate 17Hz notch filter
interpolated_psd = np.zeros_like(psd)
for iSub in range(115):
    f, interpolated_psd[iSub] = interpolate_spectra(f, psd[iSub], [16, 18], buffer=1)

np.save(f"{outdir}/interp_psd.npy", interpolated_psd)


# Visualise Interpolation
mean_psd = np.mean(psd,axis=(0,1))
mean_interp_psd = np.mean(interpolated_psd,axis=(0,1))

fig,ax = plt.subplots(dpi=300)
ax.plot(f,mean_psd)
ax.plot(f,mean_interp_psd,color='k', linestyle='--',linewidth=.8)
ax.set_xlabel('Frequency (Hz)',fontsize=14,labelpad=7)
ax.set_ylabel('Power (a.u.)',fontsize=14,labelpad=7)
ax.yaxis.set_major_locator(plt.MaxNLocator(5))
ax.legend(['PSD','Interpolated PSD'],frameon=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.savefig('.../RBD_x_PD/Glasser52/rest/results/static/spectra/interpolation.png')

