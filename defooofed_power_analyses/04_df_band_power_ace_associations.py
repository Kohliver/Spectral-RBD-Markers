#!/usr/bin/env python3
"""
Calculate associations between band power metrics and ACE scores for each separately.
ACE scores are reduced via PCA into a single score.
A single GLM with different age predictors per group is used to assess significant associations.
Significance is calculate via permutation testing and maximum t-statistic pooling across metrics and parcels.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import glmtools as glm

#%% Load data

# Set Dirs
metric_dir = '.../RBD_x_PD/Glasser52/rest/data/static/metrics'
plot_dir = '.../RBD_x_PD/Glasser52/rest/results/static/metrics/df_band_power/associations/ace'
outdir = '.../RBD_x_PD/Glasser52/rest/data/static/df_band_power/associations/ace'

# Load Coe
pkf = np.load(f"{metric_dir}/defooofed_band_power.npy")[:,1:]

# get group list
demo = pd.read_csv('.../RBD_x_PD/Glasser52/rest/data/demographics/demographics_rest.csv')
group_list = demo['Group'].values

# Get only participants for which we have all the demographics + Remove participant with wrong Education value
nan_fooof_outl = np.zeros(len(group_list))
nan_fooof_outl[36] = 1
nan_mask = demo[['Age','Sex','Education','ACE-Total']].isna().any(axis=1).values
outl_mask = (demo[['Education']] > 39).values.squeeze()
#mask = np.logical_or(nan_mask,outl_mask)

mask = np.sum(np.vstack([nan_fooof_outl,nan_mask,outl_mask]),axis=0) > 0


# PCA to extract first priniple component across Metrics
df = demo[['ACE-Attention','ACE-Memory', 'ACE-Fluency', 
           'ACE-Language', 'ACE-Visuospatioal',]]
          # 'AMI-Behavioral', 'AMI-Social', 'AMI-Emotional',]]

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

# pca.explained_variance_ratio_
# pca.explained_variance_ratio_.cumsum()

# # Get Factor Loadings
# mapping = pca.components_[:3,:].T

# plt.plot(mapping)
# plt.legend(['1','2','3'])

#%% Run GLMs

age = demo["Age"].values[~mask]
sex = demo['Sex'].values[~mask]
education = demo['Education'].values[~mask]
ace = PC1
#ace = demo['ACE-Total'].values[~mask]
group_list = demo["Group"].values[~mask]

# Prepare regressors
# ACE vars
ace_hc1 = np.zeros(demo[~mask].shape[0])
ace_pd = np.zeros(demo[~mask].shape[0])
ace_hc2 = np.zeros(demo[~mask].shape[0])
ace_rbd = np.zeros(demo[~mask].shape[0])

ace_hc1[group_list==1] = stats.zscore(ace[group_list==1],axis=0)
ace_pd[group_list==2] = stats.zscore(ace[group_list==2],axis=0)
ace_hc2[group_list==3] = stats.zscore(ace[group_list==3],axis=0)
ace_rbd[group_list==4] = stats.zscore(ace[group_list==4],axis=0)

# Create GLM dataset
data = glm.data.TrialGLMData(
    data=pkf[~mask],
    category_list=group_list,
    age=age,
    sex = sex,
    education=education,
    ace_hc1 = ace_hc1,#[~mask],
    ace_pd = ace_pd,#[~mask],
    ace_hc2 = ace_hc2,#[~mask],
    ace_rbd = ace_rbd,#[~mask],
    dim_labels=["Subjects", "Frequencies", "Parcels"],
)

# Design matrix
DC = glm.design.DesignConfig()
DC.add_regressor(name="HC$_{PD}$", rtype="Categorical", codes=1)
DC.add_regressor(name="PD", rtype="Categorical", codes=2)
DC.add_regressor(name="HC$_{RBD}$", rtype="Categorical", codes=3)
DC.add_regressor(name="RBD", rtype="Categorical", codes=4)
DC.add_regressor(name="Age", rtype="Parametric", datainfo="age", preproc="z")
DC.add_regressor(name="Sex", rtype="Parametric", datainfo="sex", preproc="z")
DC.add_regressor(name="Education", rtype="Parametric", datainfo="education", preproc="z")
DC.add_regressor(name="ACE-HC$_{PD}$", rtype="Parametric", datainfo="ace_hc1", preproc="none")
DC.add_regressor(name="ACE-PD", rtype="Parametric", datainfo="ace_pd", preproc="none")
DC.add_regressor(name="ACE-HC$_{RBD}$", rtype="Parametric", datainfo="ace_hc2", preproc="none")
DC.add_regressor(name="ACE-RBD", rtype="Parametric", datainfo="ace_rbd", preproc="none")

DC.add_contrast(name="ACE-HC$_{all}$", values=[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0])
DC.add_contrast(name="ACE-PD", values=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
DC.add_contrast(name="ACE-RBD", values=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

design = DC.design_from_datainfo(data.info)
design.plot_summary(savepath=f"{plot_dir}/glm_design.svg", show=False)
design.plot_leverage(savepath=f"{plot_dir}/glm_leverage.svg", show=False)
design.plot_efficiency(savepath=f"{plot_dir}/glm_efficiency.svg", show=False)

# Fit the GLM
model = glm.fit.OLSModel(design, data)

def do_stats(contrast_idx, metric="tstats"):
    # Max-stat permutations
    perm = glm.permutations.MaxStatPermutation(
        design=design,
        data=data,
        contrast_idx=contrast_idx,
        nperms=10000,
        metric=metric,
        tail=0,  # two-tailed t-test
        pooled_dims=(1,2),  # pool over metrics and parcels
        nprocesses=16,
    )
    null_dist = perm.nulls

    # Calculate p-values
    if metric == "tstats":
        tstats = abs(model.tstats[contrast_idx])
        percentiles = stats.percentileofscore(null_dist, tstats)
    elif metric == "copes":
        copes = abs(model.copes[contrast_idx])
        percentiles = stats.percentileofscore(null_dist, copes)
    pvalues = 1 - percentiles / 100

    return pvalues

for i in range(model.tstats.shape[0]):
    tstats = model.tstats[i]
    pvalues = do_stats(contrast_idx=i)
    print(pvalues < .05)
    np.save(f"{outdir}/contrast_{i}.npy", tstats)
    np.save(f"{outdir}/contrast_{i}_pvalues.npy", pvalues)
