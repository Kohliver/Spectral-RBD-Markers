"""
Create overview plot for associations between ACE scores and band power.
This figure is used for Figure 3 in the manuscript.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind

import seaborn as sns
from matplotlib.lines import Line2D

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import sys
sys.path.append(".../RBD_x_PD/Glasser52/scripts/helpers/")
from plotting import prepare_plotting

# Set Dirs
metric_dir = '.../RBD_x_PD/Glasser52/rest/data/static/metrics'
t_dir = '.../RBD_x_PD/Glasser52/rest/data/static/df_band_power/associations/ace'
exp_t_dir = '.../RBD_x_PD/Glasser52/rest/data/static/aperiodic/associations/ace'
outdir = '.../RBD_x_PD/Glasser52/rest/results/static/metrics/df_band_power/associations/ace'

# get group list
demo = pd.read_csv('.../RBD_x_PD/Glasser52/rest/data/demographics/demographics_rest.csv')
group_list = demo['Group'].values

# Get only participants for which we have all the demographics
nan_fooof_outl = np.zeros(len(group_list))
nan_fooof_outl[36] = 1
nan_mask = demo[['Age','Sex','Education','ACE-Total']].isna().any(axis=1).values
outl_mask = (demo[['Education']] > 39).values.squeeze()
#mask = np.logical_or(nan_mask,outl_mask)

mask = np.sum(np.vstack([nan_fooof_outl,nan_mask,outl_mask]),axis=0) > 0

# --- PCA to extract first priniple component across Metrics ---
df = demo[['ACE-Attention','ACE-Memory','ACE-Fluency', 
           'ACE-Language', 'ACE-Visuospatioal']]
df = df.values[~mask]

# Bring data to same scale
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)

# Calculate PCA keeping components explaining 95% of the variance in data
pca = PCA(.95)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)

# Extract PC1
PC1 = x_pca[:,0]

# --- Load Other Data ---

# Load Theta
theta = np.load(f"{metric_dir}/defooofed_band_power.npy")[:,1]
theta = theta[~mask]

# Load Aperiodic Exponent
ap = np.load(f"{metric_dir}/aperiodic_exponent.npy")
ap = ap[~mask]


#%% Plotting
CoE_labels, group_colors, group_cmaps, _ = prepare_plotting()
iGroup = 4

# --- Set Up Figure ---
fig = plt.figure(dpi=600, figsize=(6,10), constrained_layout=False)
gs1 = fig.add_gridspec(nrows=2, ncols=1, wspace=.1, hspace=.2)

ax = np.zeros(3, dtype=object)
ax[1] = fig.add_subplot(gs1[0, 0])
ax[0] = fig.add_subplot(gs1[1, 0])

# ---- RBD ----

# Load tstats and find parcel with largest association
ts = np.load(f"{t_dir}/contrast_4.npy")[0]
ps = np.load(f"{t_dir}/contrast_4_pvalues.npy")[0]
parc = np.argmax(abs(ts))

t = ts[parc]
p = ps[parc]

ace_in = PC1[group_list[~mask] == iGroup]
CoE_in = theta[group_list[~mask] == iGroup]

# Plotting
sns.regplot(x=ace_in, y=CoE_in[:,parc], scatter=True, ax=ax[0], color=group_colors[iGroup-1], 
            line_kws={'linewidth':3},scatter_kws={'s':60})

# Set Make axes pretty
#ax.set_ylim([11,21])
#ax.set_yticks([0,.2,.4,.6])
#ax.set_xticks([-1,0,2])
plt.locator_params(axis='y', nbins=4)

ax[0].tick_params(axis='x', labelsize= 14)
ax[0].tick_params(axis='y', labelsize= 14)

#ax[0].set_xlabel('ACE PC1', fontsize = 16, labelpad = 10)
ax[0].set_xlabel('ACE PC1', fontsize = 18, labelpad = 10)
ax[0].set_ylabel('Theta Power (a.u.)', fontsize = 18, labelpad = 7)

if p < .001:
    ax[0].text(.02,.98,'t = ' + str(np.round(t,2)) + ', p < 0.001',
            horizontalalignment='left',
            verticalalignment='top',
            transform = ax[0].transAxes,
            fontsize = 16)
else:
    ax[0].text(.45,.98,'t = ' + str(np.round(t,2)) + ', p = ' + str(np.round(p,3)),
            horizontalalignment='right',
            verticalalignment='top',
            transform = ax[0].transAxes,
            fontsize = 16)

# Remove Box Around Subplot
sns.despine(ax=ax[0], top=True, right=True, left=False,
        bottom=False, offset=None, trim=False)
     
    
# --- Exponent ----


# Load tstats and find parcel with largest association
ts = np.load(f"{exp_t_dir}/contrast_4.npy")
ps = np.load(f"{exp_t_dir}/contrast_4_pvalues.npy")

parc = np.argmax(abs(ts))
t = ts[parc]
p = ps[parc]

ace_in = PC1[group_list[~mask] == iGroup]
CoE_in = ap[group_list[~mask] == iGroup]

# Plotting
sns.regplot(x=ace_in, y=CoE_in[:,parc], scatter=True, ax=ax[1], color=group_colors[iGroup-1], 
            line_kws={'linewidth':3},scatter_kws={'s':60})

# Set Make axes pretty
#ax.set_ylim([11,21])
#ax.set_yticks([0,.2,.4,.6])
#ax.set_xticks([-1,0,2])
plt.locator_params(axis='y', nbins=5)

ax[1].tick_params(axis='x', labelsize= 14)
ax[1].tick_params(axis='y', labelsize= 14)

ax[1].set_xlabel('', fontsize = 18, labelpad = 10)
ax[1].set_ylabel('Aperiodic Exponent', fontsize = 18, labelpad = 7)

if p < .001:
    ax[1].text(.02,.98,'t = ' + str(np.round(t,2)) + ', p < 0.001',
            horizontalalignment='left',
            verticalalignment='top',
            transform = ax[1].transAxes,
            fontsize = 16)
else:
    ax[1].text(.48,.98,'t = ' + str(np.round(t,2)) + ', p = ' + str(np.round(p,3)),
            horizontalalignment='right',
            verticalalignment='top',
            transform = ax[1].transAxes,
            fontsize = 16)

# Remove Box Around Subplot
sns.despine(ax=ax[1], top=True, right=True, left=False,
        bottom=False, offset=None, trim=False)


#Save Fig
fig.savefig(f'{outdir}/RBD_Contrast_ace_df_band_power_and_exponent.svg', format = 'svg', bbox_inches='tight', transparent=True)
#plt.close()
