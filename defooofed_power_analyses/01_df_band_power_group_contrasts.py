"""
Defoofed band power group contrasts.
Calculate metric contrasts while accounting for confounds.
Multiple comparison correction via permutation tests and maximum t-statistic pooling.
"""

import numpy as np
import pandas as pd
from scipy import stats

import glmtools as glm
from osl_dynamics.analysis import power

# %% Load data

# Set Dirs
metric_dir = ".../RBD_x_PD/Glasser52/rest/data/static/metrics"
plot_dir = ".../RBD_x_PD/Glasser52/rest/results/static/metrics/df_band_power/glms"
outdir = (".../RBD_x_PD/Glasser52/rest/data/static/df_band_power/contrasts")

# Load Coe
bandPower = np.load(f"{metric_dir}/defooofed_band_power1.npy")[:, 1:]

# get group list
demo = pd.read_csv(
    ".../RBD_x_PD/Glasser52/rest/data/demographics/demographics_rest.csv"
)
group_list = demo["Group"].values

# Get only participants for which we have all the demographics + Remove participant with wrong Education value
nan_fooof_outl = np.zeros(len(group_list))
nan_fooof_outl[36] = 1
nan_mask = demo[["Age", "Sex"]].isna().any(axis=1).values
outl_mask = (demo[["Education"]] > 39).values.squeeze()

mask = np.sum(np.vstack([nan_fooof_outl, nan_mask, outl_mask]), axis=0) > 0

# %% Run GLMs

age = demo["Age"].values
sex = demo["Sex"].values
education = demo["Education"].values

# Create GLM dataset
data = glm.data.TrialGLMData(
    data=bandPower[~mask],
    category_list=group_list[~mask],
    age=age[~mask],
    sex=sex[~mask],
    education=education[~mask],
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
DC.add_regressor(
    name="Education", rtype="Parametric", datainfo="education", preproc="z"
)

DC.add_contrast(name="PD vs HC$_{PD}$", values=[-1, 1, 0, 0, 0, 0, 0])
DC.add_contrast(name="RBD vs HC$_{RBD}$", values=[0, 0, -1, 1, 0, 0, 0])
DC.add_contrast(name="PD vs RBD", values=[0, 1, 0, -1, 0, 0, 0])

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
    np.save(f"{outdir}/contrast_{i}.npy", tstats)
    np.save(f"{outdir}/contrast_{i}_pvalues.npy", pvalues)
