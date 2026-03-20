"""
Repeat GLMs calculating group contrasts for power at each freuqency bin.
T-statistics per frequency band a represented in Figure 4.
Significance thresholds are only calculated to aid the visulisation but are not interpreted in
the manuscript. Thresholds calculated with permutation testing and 
maximum t-statistic pooling acorss parcels and frequency bins
"""

import numpy as np
import pandas as pd
from scipy import stats

import glmtools as glm
from osl_dynamics.analysis import power

#%% Load data

# Set Dirs
metric_dir = '.../RBD_x_PD/Glasser52/rest/data/static/psd'
plot_dir = '.../RBD_x_PD/Glasser52/rest/results/static/metrics/spect/glms'
outdir = '.../RBD_x_PD/Glasser52/rest/data/static/spect/contrasts'

# Load Power
f = np.load(f"{metric_dir}/f.npy")
psd = np.load(f"{metric_dir}/interp_psd.npy")
psd = psd[:,:,f <= 35]

# get group list
demo = pd.read_csv('.../RBD_x_PD/Glasser52/rest/data/demographics/demographics_rest.csv')
group_list = demo['Group'].values

# Get only participants for which we have all the demographics + Remove participant with wrong Education value
nan_mask = demo[['Age','Sex']].isna().any(axis=1).values
outl_mask = (demo[['Education']] > 39).values.squeeze()
mask = np.logical_or(nan_mask,outl_mask)

#%% Run GLMs

age = demo["Age"].values
sex = demo['Sex'].values
education = demo['Education'].values

# Create GLM dataset
data = glm.data.TrialGLMData(
    data=psd[~mask],
    category_list=group_list[~mask],
    age=age[~mask],
    sex = sex[~mask],
    education=education[~mask],
    dim_labels=["Subjects", "Frequencies", "Parcels"],
)

# Design matrix
DC = glm.design.DesignConfig()
DC.add_regressor(name="HC1", rtype="Categorical", codes=1)
DC.add_regressor(name="PD", rtype="Categorical", codes=2)
DC.add_regressor(name="HC2", rtype="Categorical", codes=3)
DC.add_regressor(name="RBD", rtype="Categorical", codes=4)
DC.add_regressor(name="Age", rtype="Parametric", datainfo="age", preproc="z")
DC.add_regressor(name="Sex", rtype="Parametric", datainfo="sex", preproc="z")
DC.add_regressor(name="Education", rtype="Parametric", datainfo="education", preproc="z")

DC.add_contrast(name="Age", values=[0, 0, 0, 0, 1, 0, 0])
DC.add_contrast(name="PD-HC", values=[-1, 1, 0, 0, 0, 0, 0])
DC.add_contrast(name="RBD-HC", values=[0, 0, -1, 1, 0, 0, 0])
DC.add_contrast(name="PD-RBD", values=[0, 1, 0, -1, 0, 0, 0])

design = DC.design_from_datainfo(data.info)
design.plot_summary(savepath=f"{plot_dir}/glm_design.png", show=False)
design.plot_leverage(savepath=f"{plot_dir}/glm_leverage.png", show=False)
design.plot_efficiency(savepath=f"{plot_dir}/glm_efficiency.png", show=False)

# Fit the GLM
model = glm.fit.OLSModel(design, data)

def do_stats(contrast_idx, metric="tstats"):
    # Max-stat permutations
    perm = glm.permutations.MaxStatPermutation(
        design=design,
        data=data,
        contrast_idx=contrast_idx,
        nperms=1000,
        metric=metric,
        tail=0,  # two-tailed t-test
        pooled_dims=(1,2),  # pool over frequencies and channels
        nprocesses=6,
        # tstat_args = {'hat_factor': 5e-3, 'varcope_smoothing': 'medfilt',
        #               'window_size': 15, 'smooth_dims': 1}
    )
    null_dist = perm.nulls

    # Calculate p-values
    if metric == "tstats":
        tstats = abs(model.tstats[contrast_idx])
        percentiles = stats.percentileofscore(null_dist, tstats)
        thresh = perm.get_thresh([95,99])
    elif metric == "copes":
        copes = abs(model.copes[contrast_idx])
        percentiles = stats.percentileofscore(null_dist, copes)
        thresh = perm.get_thresh([95,99])
    pvalues = 1 - percentiles / 100

    return pvalues, thresh

for i in range(model.tstats.shape[0]):
    tstats = model.tstats[i]
    pvalues, thresh = do_stats(contrast_idx=i)
    np.save(f"{outdir}/contrast_{i}.npy", tstats)
    np.save(f"{outdir}/contrast_{i}_thresh.npy", thresh)
    np.save(f"{outdir}/contrast_{i}_pvalues.npy", pvalues)
