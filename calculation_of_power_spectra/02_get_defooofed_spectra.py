#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 00:22:59 2023

@author: okohl

Get Defooof PSDs and store aperiodic component slope per participant.
"""

import numpy as np
import fooof

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# Set dirs
indir = '.../RBD_x_PD/Glasser52/rest/data/static/psd'
metric_dir = '.../RBD_x_PD/Glasser52/rest/data/static/metrics'

# Load Interpolated Spectra and f
f = np.load(f"{indir}/f.npy")
psd = np.load(f"{indir}/interp_psd.npy")

# Numbers
nSub = psd.shape[0]
nParc = psd.shape[1]
nfreq = f[np.logical_and(f>=2,f<=40)].shape[0]

# Run FOOOF and get defoofed spectra
aperiodic_exponent = np.zeros([nSub,nParc])
aperiodic_psd = np.zeros([nSub,nParc,nfreq])
defooofed_psd = np.zeros([nSub,nParc,nfreq])
full_psd_unlog = np.zeros([nSub,nParc,nfreq])
aperiodic_psd_unlog = np.zeros([nSub,nParc,nfreq])
defooofed_psd_unlog = np.zeros([nSub,nParc,nfreq])
has_pks = np.zeros([nSub,nParc])
rsq = np.zeros([nSub,nParc])
err = np.zeros([nSub,nParc])
offsets = np.zeros([nSub,nParc])

for iSub in range(nSub): # loop across sub
    print(f'Running Sub{iSub}')

    for iParc in range(nParc):
        fm = fooof.FOOOF(max_n_peaks=5, min_peak_height=0.05, peak_width_limits=[1,12])
        fm.fit(freqs=f, power_spectrum=psd[iSub,iParc],freq_range=[2,40])
        
        defooofed_f = fm.freqs
        aperiodic_exponent[iSub,iParc] = fm.aperiodic_params_[1]
        aperiodic_psd[iSub,iParc] = fm._ap_fit
        defooofed_psd[iSub,iParc] = fm.power_spectrum - fm._ap_fit
        full_psd_unlog[iSub,iParc] = fm.get_model('full', space='linear')
        aperiodic_psd_unlog[iSub,iParc] = fm.get_model('aperiodic',  space='linear')
        defooofed_psd_unlog[iSub,iParc] = fm.get_model('full', space='linear') - fm.get_model('aperiodic',  space='linear')
        has_pks[iSub,iParc] = fm.n_peaks_
        
        rsq[iSub,iParc] = fm.r_squared_
        err[iSub,iParc] = fm.error_
        offsets[iSub,iParc] = fm.aperiodic_params_[0]

# identify participants with no oscillatory peaks
sub = np.where(np.sum(has_pks == 0,axis=1))[0][0]
parc = np.sum(has_pks == 0,axis=1)[sub]

# for i,s in enumerate(sub):
#     for p in parc[i]:
#         print(f"Subject {s}: {p} Parcels have no oscillatory peak.") 
   
np.save(f"{indir}/full_psd_unlog.npy", full_psd_unlog)
#np.save(f"{indir}/defooofed_f.npy", defooofed_f)
#np.save(f"{indir}/defooofed_psd.npy", defooofed_psd)
np.save(f"{indir}/defooofed_psd_unlog.npy", defooofed_psd_unlog)
#np.save(f"{metric_dir}/aperiodic_exponent.npy", aperiodic_exponent)
#np.save(f"{indir}/aperiodic_psd.npy", aperiodic_psd)
np.save(f"{indir}/aperiodic_psd_unlog.npy", aperiodic_psd_unlog)
#np.save(f"{indir}/fooof_rsquared.npy", rsq)
#np.save(f"{indir}/fooof_error.npy", err)
#np.save(f"{indir}/aperiodic_offsets.npy", offsets)


#%% Plot visualising how Fooof works

# Average across parcels and subs
mean_psd = np.mean(psd,axis=(0,1))

# Apply Fooof
fm = fooof.FOOOF(max_n_peaks=5, min_peak_height=0.05, peak_width_limits=[1,12])
fm.fit(freqs=f, power_spectrum=mean_psd,freq_range=[2,40])

aperiodic = fm.get_model('aperiodic',  space='linear')
full = fm.get_model('full', space='linear')
diff = full-aperiodic
fooof_f = fm.freqs

# Plotting
intensities = [.3,.6,.9]
colors = []
for intens in intensities:
    colors.append(get_cmap('Blues')(intens))

fig,ax = plt.subplots()
ax.plot(fooof_f,aperiodic,color=colors[0],linewidth=3)
ax.plot(fooof_f,diff,color=colors[1],linewidth=3)
ax.plot(fooof_f,full,color=colors[2],linewidth=3)
ax.set_xlabel('Frequency (Hz)',fontsize=14,labelpad=7)
ax.set_ylabel('Power (a.u.)',fontsize=14,labelpad=7)
ax.yaxis.set_major_locator(plt.MaxNLocator(5))
ax.legend(['Aperiodic','Periodic','PSD'],frameon=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.savefig('.../RBD_x_PD/Glasser52/rest/results/static/spectra/fooof.png')


# Quick check to check whether whole brain average shows peaks

for iSub in range(defooofed_psd.shape[0]):
    
    # Average PSD
    tmp_psd = defooofed_psd[iSub].mean(axis=0)
    
    # Plot
    fig,ax = plt.subplots()
    ax.plot(fooof_f,tmp_psd,color=colors[2],linewidth=3)
    ax.set_xlabel('Frequency (Hz)',fontsize=14,labelpad=7)
    ax.set_ylabel('Power (a.u.)',fontsize=14,labelpad=7)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_title(f'Subject{iSub}')
    ax.axhline(0,linewidth=.8,linestyle='--')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.savefig(f'.../RBD_x_PD/Glasser52/rest/results/static/spectra/defooofed_perSub/sub{iSub}_defooofed_psd.png')
    plt.close()
    
