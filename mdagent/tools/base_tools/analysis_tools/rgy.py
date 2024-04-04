from typing import Optional

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
from langchain.tools import BaseTool

from mdagent.utils import PathRegistry


class RadiusofGyration:
    def __init__(self, path_registry):
        self.path_registry = path_registry
        self.includes_top = [".h5", ".lh5", ".pdb"]

    def _grab_files(self, pdb_id: str) -> None:
        if "_" in pdb_id:
            pdb_id = pdb_id.split("_")[0]
        self.pdb_id = pdb_id
        all_names = self.path_registry._list_all_paths()
        try:
            self.pdb_path = [
                name
                for name in all_names
                if pdb_id in name and ".pdb" in name and "records" in name
            ][0]
        except IndexError:
            raise ValueError(f"No pdb file found for {pdb_id}")
        try:
            self.dcd_path = [
                name
                for name in all_names
                if pdb_id in name and ".dcd" in name and "records" in name
            ][0]
        except IndexError:
            self.dcd_path = None
            pass
        return None

    def _load_traj(self, pdb_id: str) -> None:
        self._grab_files(pdb_id)
        if self.dcd_path:
            self.traj = md.load(self.dcd_path, top=self.pdb_path)
        else:
            self.traj = md.load(self.pdb_path)
        return None

    def rad_gyration_per_frame(self, pdb_id: str) -> str:
        self._load_traj(pdb_id)
        rg_per_frame = md.compute_rg(self.traj)

        self.rgy_file = f"files/radii_of_gyration_{self.pdb_id}.csv"

        np.savetxt(
            self.rgy_file, rg_per_frame, delimiter=",", header="Radius of Gyration (nm)"
        )
        self.path_registry.map_path(
            f"radii_of_gyration_{self.pdb_id}",
            self.rgy_file,
            description=f"Radii of gyration per frame for {self.pdb_id}",
        )
        return f"Radii of gyration saved to {self.rgy_file}"

    def rad_gyration_average(self, pdb_id: str) -> str:
        _ = self.rad_gyration_per_frame(pdb_id)
        rg_per_frame = np.loadtxt(self.rgy_file, delimiter=",", skiprows=1)
        avg_rg = rg_per_frame.mean()

        return f"Average radius of gyration: {avg_rg:.2f} nm"

    def plot_rad_gyration(self, pdb_id: str) -> str:
        _ = self.rad_gyration_per_frame(pdb_id)
        rg_per_frame = np.loadtxt(self.rgy_file, delimiter=",", skiprows=1)
        plot_name = f"{self.pdb_id}_rgy.png"

        plt.plot(rg_per_frame)
        plt.xlabel("Frame")
        plt.ylabel("Radius of Gyration (nm)")
        plt.title(f"{pdb_id} - Radius of Gyration Over Time")

        plt.savefig(plot_name)
        self.path_registry.map_path(
            f"{self.pdb_id}_radii_of_gyration_plot",
            plot_name,
            description=f"Plot of radii of gyration over time for {self.pdb_id}",
        )
        return "Plot saved as: " + f"{plot_name}.png"


class RadiusofGyrationAverage(BaseTool):
    name = "RadiusofGyrationAverage"
    description = """This tool calculates the average radius of gyration
    for the given trajectory file. Give this tool the
    protein ID (PDB ID) only. The tool will automatically find the necessary files."""

    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, pdb_id: str) -> str:
        """use the tool."""
        try:
            RGY = RadiusofGyration(self.path_registry)
            return "Succeeded. " + RGY.rad_gyration_average(pdb_id)
        except ValueError as e:
            return f"Failed. ValueError: {e}"
        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


class RadiusofGyrationPerFrame(BaseTool):
    name = "RadiusofGyrationPerFrame"
    description = """This tool calculates the radius of gyration
    at each frame of a given trajectory file. Give this tool the
    protein ID (PDB ID) only. The tool will automatically find the necessary files.
    The tool will save the radii of gyration to a csv file and
    map it to the registry."""

    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, pdb_id: str) -> str:
        """use the tool."""
        try:
            RGY = RadiusofGyration(self.path_registry)
            return "Succeeded. " + RGY.rad_gyration_per_frame(pdb_id)
        except ValueError as e:
            return f"Failed. ValueError: {e}"
        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


class RadiusofGyrationPlot(BaseTool):
    name = "RadiusofGyrationPlot"
    description = """This tool calculates the radius of gyration
    at each frame of a given trajectory file and plots it.
    Give this tool the protein ID (PDB ID) only.
    The tool will automatically find the necessary files.
    The tool will save the plot to a png file and map it to the registry."""

    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, pdb_id: str) -> str:
        """use the tool."""
        try:
            RGY = RadiusofGyration(self.path_registry)
            return "Succeeded. " + RGY.plot_rad_gyration(pdb_id)
        except ValueError as e:
            return f"Failed. ValueError: {e}"
        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
