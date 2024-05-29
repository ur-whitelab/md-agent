import os
from typing import Optional

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from mdagent.utils import FileType, PathRegistry


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
    file_path = f"{path_registry.ckpt_records}/{file_id}.csv"
    i = 0
    while os.path.exists(file_path):
        i += 1
        file_path = f"{path_registry.ckpt_records}/{file_id}_{i}.csv"
    np.savetxt(file_path, data, delimiter=",")
    path_registry.map_path(file_id, file_path, description=description)
    return file_path


class MOIFunctions:
    def __init__(self, path_registry, top_fileid, traj_fileid=None, mol_name=None):
        self.path_registry = path_registry
        self.moments_of_inertia = None
        self.min_moi = None
        self.avg_moi = None
        self.mol_name = None
        if mol_name is None:
            self.mol_name = top_fileid.replace("top_", "")
        self.traj = load_traj(self.path_registry, top_fileid, traj_fileid)

    def calculate_moment_of_inertia(self):
        """
        Calculate principal moments of inertia for a molecule or protein.
        Expected shape of moments_of_inertia: (n_frames, 3)
        """
        inertia_tensors = md.compute_inertia_tensor(self.traj)
        principal_moments = np.empty((len(inertia_tensors), 3))
        for i, tensor in enumerate(inertia_tensors):
            eigenvalues, _ = np.linalg.eigh(tensor)
            principal_moments[i] = eigenvalues

        self.moments_of_inertia = principal_moments
        self.min_moi = np.min(self.moments_of_inertia)  # min of all frames & moments
        self.avg_moi = np.mean(
            self.moments_of_inertia
        )  # average of all frames & moments

        # save to file
        file_id = f"MOI_{self.mol_name}"
        description = (f"Moments of inertia tensor for {self.mol_name}",)
        csv_path = save_to_csv(
            self.path_registry, self.moments_of_inertia, file_id, description
        )
        message = (
            f"Average Moment of Inertia Tensor: {self.avg_moi}, "
            f"Data saved to: {csv_path} with file ID {file_id}"
        )
        return message

    def plot_moi(self):
        """
        Analyze and visualize the principal moments of inertia.
        """
        message = ""
        if self.moments_of_inertia is None:
            message += self.calculate_moment_of_inertia()

        if len(self.moments_of_inertia) == 1:  # only one frame
            message += "Only one frame in trajectory, no plot generated."
            return message

        fig_analysis = f"MOI_{self.mol_name}"
        fig_name = self.path_registry.write_file_name(
            type=FileType.FIGURE,
            fig_analysis=fig_analysis,
            file_format="png",
        )
        fig_id = self.path_registry.get_fileid(file_name=fig_name, type=FileType.FIGURE)

        plt.plot(self.avg_moi, label="$I_avg$", linestyle="--", color="black")
        plt.plot(self.min_moi, label="$I_min$", linestyle="--", color="red")
        plt.plot(self.moments_of_inertia[:, 0], label="$I_1$")  # smallest MOI
        plt.plot(self.moments_of_inertia[:, 1], label="$I_2$")
        plt.plot(self.moments_of_inertia[:, 2], label="$I_3$")  # largest MOI
        plt.xlabel("Frame")
        plt.ylabel("Moments of Inertia")
        plt.title("Moments of Inertia over Time")
        plt.legend()
        plt.savefig(f"{self.path_registry.ckpt_figures}/{fig_name}")
        plt.close()
        self.path_registry.map_path(
            fig_id,
            f"{self.path_registry.ckpt_figures}/{fig_name}",
            description=f"Plot of moments of inertia over time for {self.mol_name}",
        )
        message += (
            f"Plot of moments of inertia over time saved as: "
            f"{fig_name}.png with plot ID {fig_id}. "
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
            moi_funcs = MOIFunctions(
                self.path_registry, top_fileid, traj_fileid, molecule_name
            )
            msg = moi_funcs.plot_moi()
            return f"Succeeded. {msg}"
        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"
