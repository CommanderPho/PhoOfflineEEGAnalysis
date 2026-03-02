import time
import re
from datetime import datetime, timezone
from attrs import define, field, Factory
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from mne.channels.montage import DigMontage
from pathlib import Path
import numpy as np
import pandas as pd
import mne

# ElectrodeHelper module
from pathlib import Path
import importlib.resources as resources
from phopymnehelper.anatomy_and_electrodes import ElectrodeHelper



if __name__ == "__main__":
    # from phopymnehelper.anatomy_and_electrodes import ElectrodeHelper
    from phopymnehelper.anatomy_and_electrodes import ElectrodeHelper
    from mne.channels.montage import DigMontage
    from matplotlib import pyplot as plt
    import matplotlib.pyplot as plt

    import os.path as op

    import numpy as np

    import mne
    # from mne.channels.montage import get_builtin_montages
    # from mne.datasets import fetch_fsaverage
    # from mne.viz import set_3d_title, set_3d_view


    electrode_pos_parent_folder: Path = Path("E:/Dropbox (Personal)/Hardware/Consumer EEG Headsets/Emotiv Epoc EEG/ElectrodeLayouts").resolve()
    electrode_positions_path: Path = electrode_pos_parent_folder.joinpath('ElectrodePositions_2025-08-14', 'brainstorm_electrode_positions_PhoHAle_eeg_subjectspacemm.tsv')

    active_electrode_man: ElectrodeHelper = ElectrodeHelper.init_EpocX_montage(electrode_positions_path=electrode_positions_path)
    emotiv_epocX_montage: DigMontage = active_electrode_man.active_montage

    # Just create montage from your electrode positions
    print("Montage created successfully!")
    print(f"Channel names: {emotiv_epocX_montage.ch_names}")
    # Visualize the montage
    ElectrodeHelper.visualize_montage(emotiv_epocX_montage)


    # subjects_dir = Path(r"C:/Users/pho/Documents/MATLAB/brainstorm_database/2025-07-12_Lab_Brainstorm_Protocol01/anat/PhoHAle").resolve()
    # subjects_dir = Path(r"\\wsl.localhost\Ubuntu-22.04\home\pho\freesurf_subjects\PhoHAle\PhoHAle").resolve()
    subjects_dir = Path(r"E:/Dropbox (Personal)/personalStore/records/Health/Ann Arbor/Pho MRI 2025-06-23/EXPORTS/From_FreeSurfer/freesurf_subjects/PhoHAle").resolve()
    assert subjects_dir.exists(), f"subjects_dir: '{subjects_dir}' does not exist!"


    import mne
    # mne.viz.set_browser_backend("Matplotlib")
    mne.viz.set_browser_backend("qt")  # or "matplotlib"
    mne.set_config("MNE_BROWSER_BACKEND", "qt")  # or "matplotlib"


    # smooth_brain_mesh_path: Path = Path(r"E:/Dropbox (Personal)/personalStore/records/Health/Ann Arbor/Pho MRI 2025-06-23/EXPORTS/From_brain2print/brain1_smoother_mesh.obj").resolve()
    # verts, faces = mne.read_surface(smooth_brain_mesh_path, file_format='obj')


    # subjects_dir = mne.datasets.sample.data_path()
    brain_kwargs = dict(alpha=0.5, subjects_dir=subjects_dir,
        hemi="lh", surf="pial", size=(800, 600),
    )
    Brain = mne.viz.get_brain_class()
    brain = Brain("PhoHAle", **brain_kwargs)
    # brain.add_head(alpha=0.5)

    # subjects_dir = op.dirname(fetch_fsaverage(subjects_dir=subjects_dir))

    # for current_montage in get_builtin_montages():
    #     montage = mne.channels.make_standard_montage(current_montage)
    #     # Create dummy info
    #     info = mne.create_info(ch_names=montage.ch_names, sfreq=100.0, ch_types="eeg")
    #     info.set_montage(montage)
    #     fig = mne.viz.plot_alignment(
    #         # Plot options
    #         show_axes=True,
    #         dig="fiducials",
    #         surfaces="head",
    #         mri_fiducials=True,
    #         subject="fsaverage",
    #         subjects_dir=subjects_dir,
    #         info=info,
    #         coord_frame="mri",
    #         trans="fsaverage",  # transform from head coords to fsaverage's MRI
    #     )
    #     set_3d_view(figure=fig, azimuth=135, elevation=80)
    #     set_3d_title(figure=fig, title=current_montage)


    # Brain = mne.viz.get_brain_class()
    # brain = Brain(
    #     "sample", hemi="lh", surf="pial", subjects_dir=subjects_dir, size=(800, 600)
    # )
    brain.add_annotation("aparc.a2009s", borders=False)


    # active_electrode_man: ElectrodeHelper = ElectrodeHelper.init_EpocX_montage()
    # emotiv_epocX_montage: DigMontage = active_electrode_man.active_montage
    # emotiv_epocX_montage


    # ElectrodeHelper.visualize_montage(emotiv_epocX_montage)
    # plt.show()
    ## wait until closed by user
    while plt.fignum_exists(1):
        time.sleep(0.1) 
    print("Done")
