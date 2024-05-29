import os
from typing import Optional

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from mdagent.utils import FileType, PathRegistry


class SASAFunctions:
    def __init__(self, path_registry, top_fileid, traj_fileid=None, mol_name=None):
        """
        Initialize the SASAFunctions class with topology and/or trajectory files.

        Parameters:
        path_registry (PathRegistry): mapping file IDs to file paths.
        top_fileid (str): File ID for the topology file.
        traj_fileid (str, optional): File ID for the trajectory file.
        mol_name (str, optional): Name of the molecule or protein.
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
        self.molecule_name = mol_name if mol_name else top_fileid.replace("top_", "")
        self.sasa = None
        self.residue_sasa = None

    def calculate_sasa(self, probe_radius=0.14):
        """
        Calculate the Solvent Accessible Surface Area (SASA) for each
        frame in the trajectory using Shrake-Rupley algorithm. Hydrogens
        are excluded for SASA residues calculation.

        Parameters:
        probe_radius (float, optional): The radius of the probe used to calculate SASA.
            Default is 0.14 nm (1.4 Å).
        """
        self.sasa = md.shrake_rupley(self.traj, probe_radius=probe_radius, mode="atom")
        self.residue_sasa = md.shrake_rupley(
            self.traj, probe_radius=probe_radius, mode="residue"
        )

        # save to file
        sasa_file = f"{self.path_registry.ckpt_records}/sasa_{self.molecule_name}.csv"
        i = 0
        while os.path.exists(sasa_file):
            i += 1
            sasa_file = (
                f"{self.path_registry.ckpt_records}/sasa_{self.molecule_name}_{i}.csv"
            )
        np.savetxt(sasa_file, self.sasa, delimiter=",", header="SASA (nm²)")
        self.path_registry.map_path(
            f"sasa_{self.molecule_name}_{i}",
            sasa_file,
            description=f"SASA values for the molecule {self.molecule_name}",
        )
        return f"SASA values computed and saved to {sasa_file}"

    def plot_sasa(self):
        """
        Plot the total SASA and per-residue SASA over time.

        Returns:
        None
        """
        message = ""
        if self.sasa is None or self.residue_sasa is None:
            message += self.calculate_sasa()

        # Returns areasnp.array, shape=(n_frames, n_features)
        # if there's only one frame, don't plot
        if self.sasa.ndim == 1:
            message += " Only one frame in trajectory. No SASA plot generated."
            return message

        fig_analysis = f"sasa_{self.molecule_name}"
        fig_name = self.path_registry.write_file_name(
            type=FileType.FIGURE, fig_analysis=fig_analysis, file_format="png"
        )
        fig_id = self.path_registry.get_fileid(file_name=fig_name, type=FileType.FIGURE)
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.plot(self.sasa)
        plt.xlabel("Frame")
        plt.ylabel("Total SASA (nm²)")
        plt.title("Total SASA over Time")

        plt.subplot(122)
        plt.imshow(self.residue_sasa.T, aspect="auto", interpolation="nearest")
        plt.colorbar(label="SASA (nm²)")
        plt.xlabel("Frame")
        plt.ylabel("Residue")
        plt.title("Per-residue SASA over Time")
        plt.tight_layout()
        plt.savefig(f"{self.path_registry.ckpt_figures}/{fig_name}")
        plt.close()
        self.path_registry.map_path(
            fig_id,
            f"{self.path_registry.ckpt_figures}/{fig_name}",
            description=f"Plot of SASA over time for {self.molecule_name}",
        )
        message += (
            f"SASA analysis completed. Saved {fig_name}.png with plot ID {fig_id}"
        )
        return message


class SASAToolInput(BaseModel):
    top_fileid: str = Field(None, description="File ID for the topology file.")
    traj_fileid: Optional[str] = Field(
        None, description="File ID for the trajectory file."
    )
    mol_name: Optional[str] = Field(None, description="Name of molecule or protein.")


class SolventAccessibleSurfaceArea(BaseTool):
    name = "SolventAccessibleSurfaceArea"
    description = (
        "Compute the Solvent Accessible Surface Area (SASA) "
        "for a molecule or protein."
    )
    args_schema = SASAToolInput
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry):
        super().__init__()
        self.path_registry = path_registry

    def _run(
        self,
        top_fileid: str,
        traj_fileid: Optional[str] = None,
        mol_name: Optional[str] = None,
    ) -> str:
        try:
            sasa_analysis = SASAFunctions(
                self.path_registry, top_fileid, traj_fileid, mol_name
            )
            return f"Succeeded. {sasa_analysis.plot_sasa()}"
        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"
