import os
from typing import Optional

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from mdagent.utils import FileType, PathRegistry


class MomentOfInertiaAnalysis:
    def __init__(self, path_registry, top_fileid, traj_fileid=None, molecule_name=None):
        """
        Initialize the MomentOfInertia class with topology and/or trajectory files.

        Parameters:
        path_registry (PathRegistry): mapping file IDs to file paths.
        top_fileid (str): File ID for the topology file.
        traj_fileid (str, optional): File ID for the trajectory file.
        molecule_name (str, optional): Name of the molecule or protein.
        """
        self.path_registry = path_registry
        all_fileids = self.path_registry.list_path_names()
        if top_fileid not in all_fileids:
            raise ValueError("Topology File ID not found in path registry")
        top_path = self.path_registry.get_mapped_path(top_fileid)

        if traj_fileid:
            if traj_fileid not in all_fileids:
                raise ValueError("Trajectory File ID not found in path registry")
            traj_path = self.path_registry.get_mapped_path(traj_fileid)
            self.traj = md.load(traj_path, top=top_path)
        else:
            self.traj = md.load(top_path)
        self.center_of_mass = None
        self.moments_of_inertia = None
        self.avg_moi = None
        if molecule_name:
            self.molecule_name = molecule_name
        else:
            # use top_file, removing 'top_' if top_fileid contains 'top'
            self.molecule_name = top_fileid.replace("top_", "")

    def calculate_center_of_mass(self):
        """
        Calculate the center of mass for each frame in the trajectory.

        Returns:
        np.ndarray: The center of mass for each frame in the trajectory.
        """
        masses = np.array([atom.element.mass for atom in self.traj.topology.atoms])
        total_mass = np.sum(masses)
        self.center_of_mass = (
            np.sum(self.traj.xyz[:, :, :] * masses[np.newaxis, :, np.newaxis], axis=1)
            / total_mass
        )
        return self.center_of_mass

    def calculate_moment_of_inertia(self):
        """
        Calculate the moment of inertia tensor for each frame in the trajectory.

        Returns:
        np.ndarray: The moment of inertia tensor for each frame in the trajectory.
        """
        if self.center_of_mass is None:
            self.calculate_center_of_mass()

        masses = np.array([atom.element.mass for atom in self.traj.topology.atoms])
        inertia_tensors = []

        for i in range(self.traj.n_frames):
            r = self.traj.xyz[i, :, :] - self.center_of_mass[i, :]
            inertia_tensor = np.zeros((3, 3))

            for j, atom in enumerate(self.traj.topology.atoms):
                mass = masses[j]
                inertia_tensor[0, 0] += mass * (r[j, 1] ** 2 + r[j, 2] ** 2)
                inertia_tensor[1, 1] += mass * (r[j, 0] ** 2 + r[j, 2] ** 2)
                inertia_tensor[2, 2] += mass * (r[j, 0] ** 2 + r[j, 1] ** 2)
                inertia_tensor[0, 1] -= mass * r[j, 0] * r[j, 1]
                inertia_tensor[0, 2] -= mass * r[j, 0] * r[j, 2]
                inertia_tensor[1, 2] -= mass * r[j, 1] * r[j, 2]

            inertia_tensor[1, 0] = inertia_tensor[0, 1]
            inertia_tensor[2, 0] = inertia_tensor[0, 2]
            inertia_tensor[2, 1] = inertia_tensor[1, 2]

            inertia_tensors.append(inertia_tensor)

        self.moments_of_inertia = np.array(inertia_tensors)
        self.avg_moi = np.mean(self.moments_of_inertia, axis=0)
        return self.moments_of_inertia

    def compute_moi(self):
        """
        Compute the moment of inertia tensor for the molecule or protein.
        Saved to a file and returned as a dictionary.

        Returns:
        str: A message indicating the completion of the computation and the file id.
        """
        self.calculate_center_of_mass()
        self.calculate_moment_of_inertia()
        # save to a file another than .npy
        moi_file = f"{self.path_registry.ckpt_figures}/MOI_{self.molecule_name}.csv"
        i = 0
        while os.path.exists(moi_file):
            i += 1
            moi_file = (
                f"{self.path_registry.ckpt_figures}/MOI_{self.molecule_name}_{i}.csv"
            )
        moi_id = f"MOI_{self.molecule_name}_{i}"
        np.savetxt(moi_file, self.moments_of_inertia, delimiter=",")
        self.path_registry.map_path(
            moi_id,
            moi_file,
            description=f"Moments of inertia tensor for {self.molecule_name}",
        )
        message = (
            f"Average Moment of Inertia Tensor: {self.avg_moi}, "
            f"All Moments of Inertia Tensors saved to: "
            f"{moi_file} with file ID {moi_id}"
        )
        return message

    def analyze_moi(self):
        """
        Analyze and visualize the moment of inertia tensors.

        Returns:
        str: A message indicating the completion of the analysis and the plot id.
        """
        message = ""
        if self.moments_of_inertia is None or self.avg_moi is None:
            message += self.compute_moi()

        # Compute eigenvalues and eigenvectors for average tensor
        eigenvalues, eigenvectors = np.linalg.eigh(self.avg_moi)
        print("Principal Moments of Inertia (Average):", eigenvalues)
        print("Principal Axes of Rotation (Average):\n", eigenvectors)

        # Plot the principal moments of inertia over the trajectory
        principal_moments = []
        for tensor in self.moments_of_inertia:
            eigenvalues, _ = np.linalg.eigh(tensor)
            principal_moments.append(eigenvalues)

        principal_moments = np.array(principal_moments)
        fig_analysis = f"MOI_{self.molecule_name}"
        fig_name = self.path_registry.write_file_name(
            type=FileType.FIGURE,
            fig_analysis=fig_analysis,
            file_format="png",
        )
        fig_id = self.path_registry.get_fileid(file_name=fig_name, type=FileType.FIGURE)

        plt.plot(principal_moments[:, 0], label="I1")
        plt.plot(principal_moments[:, 1], label="I2")
        plt.plot(principal_moments[:, 2], label="I3")
        plt.xlabel("Frame")
        plt.ylabel("Principal Moments of Inertia")
        plt.title("Principal Moments of Inertia over Time")
        plt.legend()
        plt.savefig(f"{self.path_registry.ckpt_figures}/{fig_name}")
        plt.close()
        self.path_registry.map_path(
            fig_id,
            f"{self.path_registry.ckpt_figures}/{fig_name}",
            description=f"Plot of radii of gyration over time for {self.molecule_name}",
        )
        message += (
            f"Plot of principal moments of inertia over time saved as: "
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
            moi = MomentOfInertiaAnalysis(
                self.path_registry, top_fileid, traj_fileid, molecule_name
            )
            return f"Succeeded. {moi.analyze_moi()}"
        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"
