#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:23:51 2023

@author: okohl

RBD Rest
"""

from osl.maxfilter import run_maxfilter_batch


#%% Neo

print("-------------------------------------")
print('Running Maxfiltering for Neo')
print("--------------------------------------")

# Setup paths to raw (pre-maxfiltered) fif files
txt_in ='.../PD_x_RBD/RBD/rest/scripts/01_maxfilter/input_files/maxfilter_rest.txt'    
with open(txt_in) as f:
    input_files = f.read().splitlines()

# Directory to save the maxfiltered data to
output_directory = ".../PD_x_RBD/RBD/rest/data/01_maxfilter/"

# Run MaxFiltering
run_maxfilter_batch(
    input_files,
    output_directory,
    "--scanner neo --mode multistage --tsss --headpos --movecomp",
)
