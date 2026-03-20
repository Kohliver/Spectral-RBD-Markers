#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 17:21:34 2023

@author: okohl

Signal-space projection (SSP) denoising.
Resting-State Data of RBD MEG Study.

"""

import os
import mne
import matplotlib.pyplot as plt

from osl import preprocessing


# Directories
preproc_dir = ".../PD_x_RBD/RBD/rest/data/17Hz_Notch/02_toSSP/"
ssp_preproc_dir = ".../PD_x_RBD/RBD/rest/data/17Hz_Notch/03_SSP/"
report_dir = ".../PD_x_RBD/RBD/rest/data/17Hz_Notch/03_SSP/report/"

os.makedirs(ssp_preproc_dir, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)

# Subjects
subjects_nproj1 = ['HC1','HC8','HC9','HC11','HC12',
                   'HC13','HC16','HC17','HC18', 
                   "HC21","HC22","HC25", 'RBD2',
                   'RBD6','RBD7','RBD9','RBD10',
                   'RBD11','RBD14','RBD16','RBD18',
                   'RBD22','RBD23','RBD24','RBD28']

subjects_nproj2 = ['HC3','HC5','HC15',
                   "HC23",'RBD3','RBD5',
                   'RBD12','RBD19',
                   'RBD25',]

subjects_nproj3 = ['HC19','HC20','HC24','RBD4','RBD13']

subjects_nproj5 = ['HC10']


for ecg_proj, subjects in zip([1,2,3,5],[subjects_nproj1, subjects_nproj2, subjects_nproj3,subjects_nproj5]):

    # Paths to files
    preproc_files = []
    ssp_preproc_files = []
    for subject in subjects:
        preproc_files.append(f"{preproc_dir}/{subject}_resting_open_raw_tsss/{subject}_resting_open_tsss_preproc_raw.fif")
        ssp_preproc_files.append(f"{ssp_preproc_dir}/{subject}/{subject}_resting_open_preproc_raw.fif")
    
    for index in range(len(preproc_files)):
        subject = subjects[index]
        preproc_file = preproc_files[index]
        output_raw_file = ssp_preproc_files[index]
    
        # Make output directory
        os.makedirs(os.path.dirname(output_raw_file), exist_ok=True)
    
        # Load preprocessed fif and ICA
        dataset = preprocessing.read_dataset(preproc_file, preload=True)
        raw = dataset["raw"]
    
        # Only keep MEG, ECG, EOG, EMG
        raw = raw.pick_types(meg=True, ecg=True, eog=True, emg=True)
    
        # Create a Raw object without any channels marked as bad
        raw_no_bad_channels = raw.copy()
        raw_no_bad_channels.load_bad_channels()
    
        #  Calculate SSP using ECG
        n_proj = ecg_proj
        ecg_epochs = mne.preprocessing.create_ecg_epochs(
            raw_no_bad_channels, picks="all"
        ).average(picks="all")
        ecg_projs, events = mne.preprocessing.compute_proj_ecg(
            raw_no_bad_channels,
            n_grad=n_proj,
            n_mag=n_proj,
            n_eeg=0,
            no_proj=True,
            reject=None,
            n_jobs=6,
        )
    
        # Add ECG SSPs to Raw object
        raw_ssp = raw.copy()
        raw_ssp.add_proj(ecg_projs.copy())
    
        # Calculate SSP using EOG
        n_proj = 1
        eog_epochs = mne.preprocessing.create_eog_epochs(
            raw_no_bad_channels, picks="all"
        ).average()
        eog_projs, events = mne.preprocessing.compute_proj_eog(
            raw_no_bad_channels,
            n_grad=n_proj,
            n_mag=n_proj,
            n_eeg=0,
            no_proj=True,
            reject=None,
            n_jobs=6,
        )
    
        # Add EOG SSPs to Raw object
        raw_ssp.add_proj(eog_projs.copy())
    
        # Apply SSPs
        raw_ssp.apply_proj()
    
        # Plot power spectrum of cleaned data
        raw_ssp.plot_psd(fmax=45, n_fft=int(raw.info["sfreq"] * 4))
        plt.savefig(f"{report_dir}/psd_{subject}.png", bbox_inches="tight")
        plt.close()
    
        if len(ecg_projs) > 0:
            fig = mne.viz.plot_projs_joint(ecg_projs, ecg_epochs, show=False)
            plt.savefig(f"{report_dir}/proj_ecg_{subject}.png", bbox_inches="tight")
            plt.close()
    
        if len(eog_projs) > 0:
            fig = mne.viz.plot_projs_joint(eog_projs, eog_epochs, show=False)
            plt.savefig(f"{report_dir}/proj_eog_{subject}.png", bbox_inches="tight")
            plt.close()
    
        # Save cleaned data
        raw_ssp.save(output_raw_file, overwrite=True)
