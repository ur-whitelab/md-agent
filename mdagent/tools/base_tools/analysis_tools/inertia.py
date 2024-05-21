import os
from typing import Optional

import mdtraj as md
import numpy as np
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from mdagent.utils import PathRegistry

def load_traj(path_registry, top_fileid, traj_fileid=None):
    all_fileids = path_registry.list_path_names()
    if top_fileid not in all_fileids:
        raise ValueError("Topology File ID not found in path registry")
    top_path = path_registry.get_mapped_path(top_fileid)

    if traj_fileid is None:
        return md.load(top_path)
    
    if traj_fileid not in all_fileids:
        raise ValueError("Trajectory File ID not found in path registry")
    
    traj_path = path_registry.get_mapped_path(traj_fileid)
    return md.load(traj_path, top=top_path)


def save_to_csv(path_registry, data, file_id, description=None):
    file_path = f"{path_registry.ckpt_figures}/{file_id}.csv"
    i = 0
    while os.path.exists(file_path):
        i += 1
        file_path = f"{path_registry.ckpt_figures}/{file_id}_{i}.csv"
    np.savetxt(file_path, data, delimiter=",")
    path_registry.map_path(file_id, file_path, description=description)
    return file_path


def calculate_moment_of_inertia(path_registry, top_fileid, traj_fileid=None, mol_name=None):
    if mol_name is None:
        mol_name = top_fileid.replace("top_", "")
    
    traj = load_traj(path_registry, top_fileid, traj_fileid)
    moments_of_inertia = md.compute_inertia_tensor(traj)
    avg_moi = np.mean(moments_of_inertia, axis=0)

    # save to file
    file_id = f"MOI_{mol_name}"
    description=f"Moments of inertia tensor for {mol_name}",
    csv_path = save_to_csv(path_registry, moments_of_inertia, file_id, description)
    message = (
        f"Average Moment of Inertia Tensor: {avg_moi}, "
        f"Data saved to: {csv_path} with file ID {file_id}"
    )
    return message


class MomentOfInertiaToolInput(BaseModel):
    top_fileid: str = Field(None, description="File ID for the topology file.")
    traj_fileid: Optional[str] = Field(
        None, description="File ID for the trajectory file."
    )
    molecule_name: Optional[str] = Field(
        None, description="Name of the molecule or protein."
    )

class MomentOfInertia(BaseTool):
    name = "MomentOfInertia"
    description = (
        "Compute the moment of inertia tensors for a molecule or protein."
        "Give this tool file IDs for topology and trajectory files as needed."
    )
    args_schema = MomentOfInertiaToolInput
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry):
        super().__init__()
        self.path_registry = path_registry

    def _run(
        self,
        top_fileid: str,
        traj_fileid: Optional[str] = None,
        molecule_name: Optional[str] = None,
    ) -> str:
        try:
            msg = calculate_moment_of_inertia(
                self.path_registry, top_fileid, traj_fileid, molecule_name
            )
            return f"Succeeded. {msg}"
        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"
