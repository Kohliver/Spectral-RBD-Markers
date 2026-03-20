"""
Correlation plots for parcels with the highest t-statistic for significant associations between 
band power and ace scores.
PCA results and group contrasts of PC1 between the groups are also visualised.
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
outdir = '.../RBD_x_PD/Glasser52/rest/results/static/metrics/df_band_power/associations/ace'

# get group list
demo = pd.read_csv('/ohba/pi/knobre/okohl/RBD_x_PD/Glasser52/rest/data/demographics/demographics_rest.csv')
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
ax.bar([1,2,3,4,5],explained_var,width=.7)

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
fig.savefig(f'{outdir}/ACE_ExplainedVar.svg', format = 'svg', bbox_inches='tight', transparent=True)


# --- Plot Factor Loadings ---

fig, ax = plt.subplots(dpi=300)
plt.bar(range(1,6),loadings[:,0],width=.7)

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
CoE = np.load(f"{metric_dir}/defooofed_band_power.npy")[:,0]
CoE = CoE[~mask]

# Load tstats and find parcel with largest association
ts = np.load(f"{t_dir}/contrast_4.npy")[0]
ps = np.load(f"{t_dir}/contrast_4_pvalues.npy")[0]
parc = np.argmax(abs(ts))

t = ts[parc]
p = ps[parc]

CoE_labels, group_colors, group_cmaps, _ = prepare_plotting()

iGroup = 4

ace_in = PC1[group_list[~mask] == iGroup]
CoE_in = CoE[group_list[~mask] == iGroup]

# Set up Figure
fig, ax = plt.subplots(dpi=300)
sns.regplot(x=ace_in, y=CoE_in[:,parc], scatter=True, ax=ax, color=group_colors[iGroup-1])

# Set Make axes pretty
#ax.set_ylim([11,21])
#ax.set_yticks([0,.2,.4,.6])
#ax.set_xticks([-1,0,2])
plt.locator_params(axis='y', nbins=5)

ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

ax.set_xlabel('ACE PC1', fontsize = 16, labelpad = 10)
ax.set_ylabel('Theta Power (a.u.)', fontsize = 16, labelpad = 7)

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
fig.savefig(f'{outdir}/RBD_Contrast_ace_df_band_power_parc{parc}.svg', format = 'svg', bbox_inches='tight', transparent=True)
plt.close()
     
     
    
#%% All

# Load tstats and find parcel with largest association
ts = np.load(f"{t_dir}/contrast_5.npy")[0]
ps = np.load(f"{t_dir}/contrast_5_pvalues.npy")[0]

parc = np.argmax(abs(ts))
t = ts[parc]
p = ps[parc]

# Load Band Power
bp = np.load(f"{metric_dir}/band_power.npy")[:,0]
bp_in = CoE[~mask]
ace_in = PC1

bp_labels, group_colors, group_cmaps, _ = prepare_plotting()

# Set up Figure
fig, ax = plt.subplots(dpi=300)

for iGroup in range(1,5):
    x = ace_in[group_list[~mask] == iGroup]
    y = bp_in[group_list[~mask] == iGroup,parc]
    sns.scatterplot(x=x, y=y, s=60, color=group_colors[iGroup-1],ax=ax)
sns.regplot(x=ace_in, y=bp_in[:,parc], scatter=False, ax=ax, color=group_colors[0])

# Set Make axes pretty
plt.locator_params(axis='y', nbins=5)

ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

ax.set_xlabel('ACE PC1', fontsize = 16, labelpad = 10)
ax.set_ylabel('Theta Power (a.u.)', fontsize = 16, labelpad = 9)
 
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
fig.savefig(f'{outdir}/allContrast_ace_band_power_parc{parc}.svg', format = 'svg', bbox_inches='tight', transparent=True)
plt.close()
 

#%% Plot PC1 Group Differeces

# Get Group Inds
groups = group_list[~mask]

# RBD vs HC Ttests
hc2=PC1[groups==3]
rbd=PC1[groups==4]

t,p =ttest_ind(hc2,rbd)

# prepare data
df = np.vstack([PC1,groups]).T
df = pd.DataFrame(df, columns=['ace','group'])

# Get Colors
labels, cols, _, _ = prepare_plotting()
          
# plot
fig = plt.figure(dpi=600)
ax = fig.add_subplot()
points = sns.swarmplot(data=df, x="group", y="ace", palette=sns.color_palette(cols,4), size=6, alpha=1, ax=ax, legend=False)
sns.boxplot(data=df, x="group", y="ace", width = .5,
            color='white', ax=ax)


ax.set_xticklabels(labels)
ax.set_xlabel('')
ax.set_yticks([-2,0,2,4])
ax.set_ylabel('ACE-PC1', fontsize=16)
ax.tick_params(axis='x', which='major', labelsize=12)
ax.tick_params(axis='y', which='major', labelsize=12) 

ax.axhline(0, color = 'grey', linestyle = '--', linewidth = .5)

# Legend
# Create a legend with reduced points
custom_lines = [Line2D([0], [0],
                       marker='o',
                       color='w',
                       label=label,
                       markerfacecolor=c,
                       markersize=12) 
                for label,c in zip(labels,cols)]
    
ax.legend(handles=custom_lines, 
          bbox_to_anchor=(1.01, 1.02),
          frameon=False,
          title='K',
          title_fontsize=14,
          facecolor='white',
          framealpha=1,
          prop={"size": 10},
          labelspacing=0.2,
          handletextpad=1,
          handlelength=1)


ax.axhline(0, color = 'grey', linestyle = '--', linewidth = .5)

# Add sig star
if p < .05:
    if p <= .01 :
        pval = '**'
    elif p <= .05:
        pval = '*'
    else:
        pval = ''
    
    # Add asterix
    x_position = 2.4
    y_position = max(PC1) * 1.12

ax.text(x=x_position, y=y_position, s=pval, zorder=10, size=16 )
ax.hlines(y=(y_position-.0000001), xmin=2, xmax=3, color='#444444', linewidth=.8)
    
# Remove Box Around Subplot
sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)

#Save Fig
fig.savefig(f'{outdir}/PC1_ACE_group_contrast.svg', format = 'svg', bbox_inches='tight', transparent=True)
plt.close()

# HC vs PD
hc1=PC1[groups==1]
pd=PC1[groups==2]

t,p =ttest_ind(hc1,pd)
