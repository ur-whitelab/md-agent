import os

import mdtraj as md
import numpy as np


def load_single_traj(path_registry, top_fileid, traj_fileid=None, traj_required=False):
    all_fileids = path_registry.list_path_names()
    if top_fileid not in all_fileids:
        raise ValueError("Topology File ID not found in path registry")
    top_path = path_registry.get_mapped_path(top_fileid)

    if traj_fileid is None:
        if not traj_required:
            return md.load(top_path)
        else:
            raise ValueError("Trajectory File ID is required, and it's not provided.")

    if traj_fileid not in all_fileids:
        raise ValueError("Trajectory File ID not found in path registry.")

    traj_path = path_registry.get_mapped_path(traj_fileid)
    return md.load(traj_path, top=top_path)


def save_to_csv(path_registry, data_to_save, file_id, description=None):
    file_path = f"{path_registry.ckpt_records}/{file_id}.csv"
    i = 0
    while os.path.exists(file_path):
        i += 1
        file_path = f"{path_registry.ckpt_records}/{file_id}_{i}.csv"
    np.savetxt(file_path, data_to_save, delimiter=",")
    path_registry.map_path(file_id, file_path, description=description)
    return file_path
