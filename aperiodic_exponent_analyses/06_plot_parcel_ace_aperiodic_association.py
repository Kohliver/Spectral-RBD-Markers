"""
Correlation plots for parcel with largest t-statistic of parcels with significant associations between
ACE scores and aperiodic exponents in the PD group.
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
t_dir = '.../RBD_x_PD/Glasser52/rest/data/static/aperiodic/associations/ace'
outdir = '.../RBD_x_PD/Glasser52/rest/results/static/metrics/aperiodic/associations/ace'

# get group list
demo = pd.read_csv('.../RBD_x_PD/Glasser52/rest/data/demographics/demographics_rest.csv')
group_list = demo['Group'].values

# Get only participants for which we have all the demographics
nan_fooof_outl = np.zeros(len(group_list))
nan_fooof_outl[36] = 1
nan_mask = demo[['Age','Sex','Education','ACE-Total']].isna().any(axis=1).values
outl_mask = (demo[['Education']] > 39).values.squeeze()
mask = np.sum(np.vstack([nan_fooof_outl,nan_mask,outl_mask]),axis=0) > 0

# PCA to extract first priniple component across Metrics
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

explained_var = pca.explained_variance_ratio_

# Get Factor Loadings
loadings = pca.components_[:3,:].T

# --- PLot Explained Variance ---

# Set up Figure
fig, ax = plt.subplots()
ax.bar([1,2,3,4,5],explained_var)

ax.set_xticklabels(['','PC1','PC2','PC3','PC4','PC5'])
ax.set_yticks([0,.1,.2,.3,.4])

ax.tick_params(axis='x', labelsize= 14)
ax.tick_params(axis='y', labelsize= 12)

ax.set_xlabel('', fontsize = 16, labelpad = 10)
ax.set_ylabel('Explained Variance', fontsize = 16, labelpad = 7)

# Remove Box Around Subplot
sns.despine(ax=ax, top=True, right=True, left=False,
        bottom=False, offset=None, trim=False)

#Save Fig
#fig.savefig(f'{outdir}/ACE_ExplainedVar.svg', format = 'svg', bbox_inches='tight', transparent=True)


# --- Plot Factor Loadings ---

fig, ax = plt.subplots(dpi=300)
plt.bar(range(1,6),loadings[:,0])

ax.set_xticklabels('')
#ax.set_xticklabels(['','Attnetion','Memory','Fluency','Language','Visuo-\nspatial'])
ax.set_yticks([0,-.2,-.4,-.6])

ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

ax.set_xlabel('ACE-Domains', fontsize = 16, labelpad = 10)
ax.set_ylabel('Factor Loading', fontsize = 16, labelpad = 10)

# Remove Box Around Subplot
sns.despine(ax=ax, top=False, right=True, left=False,
        bottom=True, offset=None, trim=False)

#Save Fig
#fig.savefig(f'{outdir}/ACE_Loadings.svg', format = 'svg', bbox_inches='tight', transparent=True)

#%% RBD

# Load Coe
ap = np.load(f"{metric_dir}/aperiodic_exponent.npy")
ap = ap[~mask]

# Load tstats and find parcel with largest association
ts = np.load(f"{t_dir}/contrast_4.npy")
ps = np.load(f"{t_dir}/contrast_4_pvalues.npy")
parc = np.argmax(abs(ts))

t = ts[parc]
p = ps[parc]

ap_labels, group_colors, group_cmaps, _ = prepare_plotting()

iGroup = 4

ace_in = PC1[group_list[~mask] == iGroup]
ap_in = ap[group_list[~mask] == iGroup]

# Set up Figure
fig, ax = plt.subplots(dpi=300)
sns.regplot(x=ace_in, y=ap_in[:,parc], scatter=True, ax=ax, color=group_colors[iGroup-1])

ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

ax.set_xlabel('ACE PC1', fontsize = 16, labelpad = 10)
ax.set_ylabel('Aperiodic Exponent', fontsize = 16, labelpad = 7)

if p < .001:
    ax.text(.02,.98,'t = ' + str(np.round(t,2)) + ', p < 0.001',
            horizontalalignment='left',
            verticalalignment='top',
            transform = ax.transAxes,
            fontsize = 14)
else:
    ax.text(.94,.98,'t = ' + str(np.round(t,2)) + ', p = ' + str(np.round(p,3)),
            horizontalalignment='right',
            verticalalignment='top',
            transform = ax.transAxes,
            fontsize = 14)

# Remove Box Around Subplot
sns.despine(ax=ax, top=True, right=True, left=False,
        bottom=False, offset=None, trim=False)

#Save Fig
fig.savefig(f'{outdir}/RBD_Contrast_ace_ap_parc{parc}.svg', format = 'svg', bbox_inches='tight', transparent=True)
plt.close()
     
