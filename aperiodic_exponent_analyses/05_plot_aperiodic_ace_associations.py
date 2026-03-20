"""
Project results of GLMs assessing association between aperiodic exponents and ace scores onto a brain surface.
This Figure is presented in Figure 3 of the manuscript.
"""

import numpy as np
import matplotlib.pyplot as plt

from osl_dynamics.analysis import power
import sys
sys.path.append(".../RBD_x_PD/Glasser52/rest/scripts/helpers/")
import old_power
from plotting import prepare_plotting

# Set Dirs
datdir = '.../RBD_x_PD/Glasser52/rest/data/static/aperiodic/associations/ace'
outdir = '.../RBD_x_PD/Glasser52/rest/results/static/metrics/aperiodic/associations/ace'

group_labels, group_colors, _ , cropped_cmaps = prepare_plotting(inverse_cmp=True)
cmaps = ['Greens_r','Greens_r',cropped_cmaps[0],cropped_cmaps[1],
         cropped_cmaps[3],cropped_cmaps[0]]


for i, name, cmap in zip(range(6),["age","education",'HC','PD','RBD','All'],cmaps):
    ts = np.load(f"{datdir}/contrast_{i}.npy")
    pvalues = np.load(f"{datdir}/contrast_{i}_pvalues.npy")
    ts[pvalues>= .05] = 0
    
    power.save(
        ts,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={
            "cmap": "RdBu_r",
            "bg_on_data": 1,
            "darkness": 1.4,
            "alpha": 1,
            "views": ["lateral", "medial"],
            "vmin": -np.max(abs(ts)),
            "vmax": np.max(abs(ts)),
        },
        filename=f"{outdir}/{name}_thresh_.png",
    )
    plt.close()

