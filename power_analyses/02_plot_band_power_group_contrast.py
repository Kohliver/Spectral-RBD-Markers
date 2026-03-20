"""
Project results of GLMs comparing band power (without removing the aperiodic component) onto a brain surface.
Figures are presented in Figure 4 of the manuscript.
"""

import numpy as np
import matplotlib.pyplot as plt

from osl_dynamics.analysis import power
from matplotlib.colors import LinearSegmentedColormap

import sys
sys.path.append(".../RBD_x_PD/Glasser52/rest/scripts/helpers/")
import old_power


# Set Dirs
datdir = '.../RBD_x_PD/Glasser52/rest/data/static/band_power/contrasts'
outdir = '.../RBD_x_PD/Glasser52/rest/results/static/metrics/band_power/glms'

# Color Maps
color = (0.01,0.65,0.08,1)
colors = [color, (1, 1, 1, 0)]
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)


for i, name in enumerate(["age","PD-HC",'RBD-HC','PD-RBD']):
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
            "vmin": -(np.max(abs(ts))),
            "vmax": np.max(abs(ts)),
        },
        filename=f"{outdir}/{name}_thresh_.png",
    )
    plt.close()
    

