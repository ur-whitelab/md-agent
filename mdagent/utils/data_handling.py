import os

import mdtraj as md
import numpy as np

from .path_registry import PathRegistry


def load_single_traj(path_registry, top_fileid, traj_fileid=None, traj_required=False):
    """
    Load a single trajectory file using mdtraj. Check for file IDs in the path registry.

    Parameters:
    path_registry (PathRegistry): mapping file IDs to file paths.
    top_fileid (str): File ID for the topology file.
    traj_fileid (str, optional): File ID for the trajectory file.
    traj_required (bool, optional): Whether the traj file is required. Default is False.

    Returns:
    mdtraj.Trajectory: Trajectory object.
    """
    if not isinstance(path_registry, PathRegistry):
        raise ValueError("path_registry must be an instance of PathRegistry.")
    all_fileids = path_registry.list_path_names()
    if top_fileid not in all_fileids:
        raise ValueError(f"Topology File ID '{top_fileid}' not found in PathRegistry")
    top_path = path_registry.get_mapped_path(top_fileid)

    if traj_fileid is None:
        if not traj_required:
            return md.load(top_path)
        else:
            raise ValueError("Trajectory File ID is required, and it's not provided.")

    if traj_fileid not in all_fileids:
        raise ValueError(
            f"Trajectory File ID '{traj_fileid}' not found in PathRegistry."
        )

    traj_path = path_registry.get_mapped_path(traj_fileid)
    return md.load(traj_path, top=top_path)


def save_to_csv(
    path_registry, data_to_save, analysis_name, description=None, header=""
):
    """
    Saves data to a csv file and maps the file ID to the file path in the path registry.

    Parameters:
    path_registry (PathRegistry): mapping file IDs to file paths.
    data_to_save (np.ndarray): Data to save to a csv file.
    analysis_name (str): Name of the analysis or data. This will be used as the file ID.
    description (str, optional): Description of the data.

    Returns:
    str: File ID for the saved data.
    """
    if not isinstance(path_registry, PathRegistry):
        raise ValueError("path_registry must be an instance of PathRegistry.")
    if not isinstance(data_to_save, np.ndarray):
        raise TypeError("data_to_save must be an instance of np.ndarray.")

    base_path = f"{path_registry.ckpt_records}/{analysis_name}"
    file_path = f"{base_path}.csv"
    i = 0
    while os.path.exists(file_path):
        i += 1
        file_path = f"{base_path}_{i}.csv"
    file_id = analysis_name if i == 0 else f"{analysis_name}_{i}"
    np.savetxt(file_path, data_to_save, delimiter=",", header=header)
    path_registry.map_path(file_id, file_path, description=description)
    print(f"Data saved to {file_path}")
    return file_id
