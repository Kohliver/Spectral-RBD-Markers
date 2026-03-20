"""Source reconstruction.

This includes beamforming, parcellation and orthogonalisation.

RBD Resting State Data.
Parcellates data into Glasser Parcellation.
"""

import os
from dask.distributed import Client

from osl import source_recon, utils

# Directories
preproc_dir = ".../PD_x_RBD/RBD/rest/data/17Hz_Notch/03_SSP/"
coreg_dir = ".../PD_x_RBD/RBD/rest/data/17Hz_Notch/04_src/"
src_dir = ".../PD_x_RBD/RBD/rest/data/17Hz_Notch/04_src/"
fsl_dir = ".../fsl/6.0.5"  # this is where FSL is installed on hbaws

# Files
preproc_file = preproc_dir + "/{subject}/{subject}_resting_open_preproc_raw.fif"  # {subject} will be replaced by the subject name

# Subjects to do
subjects = ['HC1','HC3','HC5','HC8','HC9','HC10','HC11','HC12',
            'HC13','HC15','HC16','HC17','HC18','HC19','HC20', 
            "HC21","HC22","HC23",'HC24',"HC25",'RBD2','RBD3',
            'RBD4','RBD5','RBD6','RBD7','RBD9','RBD10',
            'RBD11','RBD12','RBD13','RBD14','RBD16','RBD18',
            'RBD19','RBD22','RBD23','RBD24','RBD25','RBD28']

# Settings
config = """
    source_recon:
    - forward_model:
        model: Single Layer
    - beamform:
        freq_range: [1, 45]
        chantypes: [mag, grad]
        rank: {meg: 60}
    - parcellate:
        parcellation_file: Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz
        method: spatial_basis
        orthogonalisation: symmetric
"""

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")
    source_recon.setup_fsl(fsl_dir)

    # # Copy directory containing the coregistration
    # if not os.path.exists(src_dir):
    #     cmd = f"cp -r {coreg_dir} {src_dir}"
    #     print(cmd)
    #     os.system(cmd)

    # Get paths to files
    preproc_files = []
    for subject in subjects:
        preproc_files.append(preproc_file.format(subject=subject))


    # Source reconstruction
    source_recon.run_src_batch(
        config,
        src_dir=src_dir,
        subjects=subjects,
        preproc_files=preproc_files,
        dask_client=False,
    )
