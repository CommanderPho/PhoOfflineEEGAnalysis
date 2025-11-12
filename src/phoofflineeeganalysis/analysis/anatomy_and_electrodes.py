import time
import re
from datetime import datetime, timezone
from attrs import define, field, Factory

import uuid
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from mne.channels.montage import DigMontage
from nptyping import NDArray
from matplotlib import pyplot as plt

from pathlib import Path
import numpy as np
import pandas as pd

import mne
from mne import set_log_level
from copy import deepcopy
import mne
from mne.channels import make_dig_montage
# import nibabel as nib
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

from mne.io import read_raw

mne.viz.set_browser_backend("Matplotlib")

import matplotlib.pyplot as plt

# from ..EegProcessing import bandpower
from numpy.typing import NDArray

# ElectrodeHelper module
from pathlib import Path
# import trimesh
import importlib.resources as resources


set_log_level("WARNING")

@define(slots=False)
class ElectrodeHelper:
    """
    A helper class for creating MNE montages from Emotiv electrode data
    and projecting them onto scalp surfaces from MRI data.
    
    
    Basic Electrode Positions Loading:
        from phoofflineeeganalysis.analysis.anatomy_and_electrodes import ElectrodeHelper

        # Just create montage from your electrode positions
        # electrode_positions_path = Path(r"E:/Dropbox (Personal)/Hardware/Consumer EEG Headsets/Emotiv Epoc EEG/ElectrodeLayouts/emotiv_wellAlignedPho.ced").resolve()
        electrode_positions_path = Path(r"E:/Dropbox (Personal)/Hardware/Consumer EEG Headsets/Emotiv Epoc EEG/ElectrodeLayouts/emotiv.ced").resolve()
        assert electrode_positions_path.exists()
        montage = ElectrodeHelper.create_complete_montage_workflow(electrode_positions_path)
        montage
        
    Stateful Electrode Positions Loading:
    
        from mne.channels.montage import DigMontage
        from phoofflineeeganalysis.analysis.anatomy_and_electrodes import ElectrodeHelper

        active_electrode_man: ElectrodeHelper = ElectrodeHelper.init_EpocX_montage()
        emotiv_epocX_montage: DigMontage = active_electrode_man.active_montage
        emotiv_epocX_montage

        # Just create montage from your electrode positions
        print("Montage created successfully!")
        print(f"Channel names: {emotiv_epocX_montage.ch_names}")
        

        
    """
    active_montage: DigMontage = field()

    
    @classmethod
    def init_EpocX_montage(cls, electrode_positions_path: Optional[Path] = None) -> "ElectrodeHelper":
        if electrode_positions_path is None:
            # electrode_pos_parent_folder: Path = Path("E:/Dropbox (Personal)/Hardware/Consumer EEG Headsets/Emotiv Epoc EEG/ElectrodeLayouts").resolve()
            # electrode_positions_path = electrode_pos_parent_folder.joinpath('ElectrodePositions_2025-08-14', 'brainstorm_electrode_positions_PhoHAle_eeg_subjectspacemm.tsv')

            # electrode_pos_parent_folder: Path = Path("E:/Dropbox (Personal)/Hardware/Consumer EEG Headsets/Emotiv Epoc EEG/ElectrodeLayouts").resolve()
            # electrode_positions_path = electrode_pos_parent_folder.joinpath('phoofflineeeganalysis/resources/ElectrodeLayouts/brainstorm_electrode_positions_PhoHAle_eeg_subjectspacemm.tsv')

            electrode_positions_path = Path(resources.files("phoofflineeeganalysis").joinpath("resources/ElectrodeLayouts/brainstorm_electrode_positions_PhoHAle_eeg_subjectspacemm.tsv")).resolve()


        assert electrode_positions_path.exists(), f"electrode_positions_path: '{electrode_positions_path}' does not exist!"
        
        emotiv_epocX_montage: DigMontage = ElectrodeHelper.montage_from_subjece_space_mm_tsv(electrode_positions_path=electrode_positions_path)
        return cls(active_montage=emotiv_epocX_montage)
        
    
    @classmethod
    def montage_from_subjece_space_mm_tsv(cls, electrode_positions_path: Path) -> DigMontage:
        """ 
        Loads a Brainstorm Exported Electrode configuration 
        To export from Brainstorm, right click an EEGLAB channels (14) object and go to:
            File > Export to file...
            From the "Files of type:" dropdown, select "EEG: BIDS electrodes.tsv, subject space mm (*.tsv)"
            For "File name:" I used 'brainstorm_electrode_positions_PhoHAle_eeg_subjectspacemm.tsv'
            
            
        Usage:
            from mne.channels.montage import DigMontage
            from phoofflineeeganalysis.analysis.anatomy_and_electrodes import ElectrodeHelper

            electrode_pos_parent_folder = Path("E:/Dropbox (Personal)/Hardware/Consumer EEG Headsets/Emotiv Epoc EEG/ElectrodeLayouts").resolve()
            electrode_positions_path = electrode_pos_parent_folder.joinpath('ElectrodePositions_2025-08-14', 'brainstorm_electrode_positions_PhoHAle_eeg_subjectspacemm.tsv')

            mont: DigMontage = ElectrodeHelper.montage_from_subjece_space_mm_tsv(electrode_positions_path=electrode_positions_path)
            mont
        """
        # subjectspacemm
        assert electrode_positions_path.exists()
        df = pd.read_csv(electrode_positions_path, sep="\t")  # or '\s+' if whitespace separated
        # Map column names from Brainstorm's format to MNE
        # Assuming columns: Name   X   Y   Z   Type
        df.columns = [c.strip().lower() for c in df.columns]  # normalize
        ch_pos = {}
        fid = {}
        for _, row in df.iterrows():
            name = row["name"]
            pos_m = (row["x"]/1000.0, row["y"]/1000.0, row["z"]/1000.0)  # mm → m
            if name.upper() in {"NAS","LPA","RPA"}:
                fid[name.upper()] = pos_m
            else:
                ch_pos[name] = pos_m

        mont: DigMontage = mne.channels.make_dig_montage(ch_pos=ch_pos, nasion=fid.get("NAS"), lpa=fid.get("LPA"), rpa=fid.get("RPA"), coord_frame="head")
        return mont

    @staticmethod
    def _parse_ced(ced_file_path: Path) -> Dict[str, np.ndarray]:
        """Very tolerant .ced parser: returns dict[label] = np.array([x,y,z]) (units: unknown)."""
        coords = {}
        float_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
        with open(ced_file_path, "r", encoding="utf8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("%") or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                # label is first token that is not purely numeric
                label = parts[0]
                # find numeric tokens in the rest of the line
                nums = float_re.findall(line)
                # if the first numeric occurs after the label, extract up to 3 numeric values
                if len(nums) >= 3:
                    # choose first three numeric tokens as x,y,z
                    x, y, z = map(float, nums[:3])
                    coords[label] = np.array([x, y, z], dtype=float)
                else:
                    # fallback: if no numeric triple, skip
                    continue
        return coords


    @staticmethod
    def visualize_montage(montage: mne.channels.DigMontage):
        """Simple visualization using MNE's built-in plot; opens interactive 3D viewer if available."""
        montage.plot(kind="3d")





if __name__ == "__main__":
    from phoofflineeeganalysis.analysis.anatomy_and_electrodes import ElectrodeHelper
    from mne.channels.montage import DigMontage

    import os.path as op

    import numpy as np

    import mne
    from mne.channels.montage import get_builtin_montages
    from mne.datasets import fetch_fsaverage
    from mne.viz import set_3d_title, set_3d_view


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
