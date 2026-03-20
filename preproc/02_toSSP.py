#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 07:45:10 2023

@author: okohl
"""

if __name__ == '__main__':
    from dask.distributed import Client
    import osl
    import os
    
    client = Client(threads_per_worker=1, n_workers=3)
       
    # Sub IDs to run preproc for
    subjects = ['HC1','HC3','HC5','HC8','HC9','HC10','HC11','HC12',
                'HC13','HC15','HC16','HC17','HC18','HC19','HC20', "HC21","HC22","HC23",'HC24',"HC25",
                'RBD2','RBD3','RBD4','RBD5','RBD6','RBD7','RBD9','RBD10',
                'RBD11','RBD12','RBD13','RBD14','RBD16','RBD18',
                'RBD19','RBD22','RBD23','RBD24','RBD25','RBD28']
    
    #subjects = ['HC5','HC12']
       
    # Settings
    config = """
        preproc:
        - filter: {l_freq: 0.5, h_freq: 125, method: iir, iir_params: {order: 5, ftype: butter}, picks: ['meg','eog','ecg']}
        - notch_filter: {freqs: 17 50 100}
        - resample: {sfreq: 250}
        - bad_segments: {segment_len: 500, picks: mag, significance_level: 0.1}
        - bad_segments: {segment_len: 500, picks: grad, significance_level: 0.1}
        - bad_segments: {segment_len: 500, picks: mag, mode: diff, significance_level: 0.1}
        - bad_segments: {segment_len: 500, picks: grad, mode: diff, significance_level: 0.1}
        - bad_channels: {picks: mag, significance_level: 0.1}
        - bad_channels: {picks: grad, significance_level: 0.1}
        - ica_raw: {picks: meg, n_components: 64}
        - ica_autoreject: {apply: False}
        - interpolate_bads: {}
    """  
    
    # Get Path to Maxfiltered Files
    indir = '.../PD_x_RBD/RBD/rest/data/01_maxfilter'
          
    inputs = []
    for subject in subjects:
        inputs.append(f"{indir}/{subject}_resting_open_raw_tsss.fif")     
         
    # Set and Create Outdor
    preproc_dir = '.../PD_x_RBD/RBD/rest/data/17Hz_Notch/02_toSSP/' 
    
    # Run batch preprocessing
    osl.preprocessing.run_proc_batch(
        config,
        inputs,
        outdir=preproc_dir,
        overwrite=True,
        dask_client=True,
    )
