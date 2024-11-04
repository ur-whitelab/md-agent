from typing import Optional

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
from langchain.tools import BaseTool

from mdagent.utils import FileType, PathRegistry, load_single_traj


class RadiusofGyration:
    def __init__(self, path_registry):
        self.path_registry = path_registry
        self.top_file = ""
        self.traj_file = ""
        self.traj = None

    def _load_traj(self, top_file: str, traj_file: str):
        self.traj_file = traj_file
        self.top_file = top_file
        self.traj = load_single_traj(
            path_registry=self.path_registry,
            top_fileid=top_file,
            traj_fileid=traj_file,
            traj_required=True,
        )

    def rgy_per_frame(self, force_recompute: bool = False) -> str:
        rg_per_frame = md.compute_rg(self.traj)
        self.rgy_file = (
            f"{self.path_registry.ckpt_figures}/radii_of_gyration_{self.traj_file}.csv"
        )
        rgy_id = f"rgy_{self.traj_file}"
        if rgy_id in self.path_registry.list_path_names() and force_recompute is False:
            print("RGY already computed, skipping re-compute")
            # todo -> maybe allow re-compute & save under different id/path
        else:
            np.savetxt(
                self.rgy_file,
                rg_per_frame,
                delimiter=",",
                header="Radius of Gyration (nm)",
            )
            self.path_registry.map_path(
                f"rgy_{self.traj_file}",
                self.rgy_file,
                description=f"Radii of gyration per frame for {self.traj_file}",
            )
        return f"Radii of gyration saved to {self.rgy_file} with id {rgy_id}."

    def rgy_average(self) -> str:
        _ = self.rgy_per_frame()
        rg_per_frame = np.loadtxt(self.rgy_file, delimiter=",", skiprows=1)
        avg_rg = rg_per_frame.mean()

        return f"Average radius of gyration: {avg_rg:.2f} nm"

    def plot_rgy(self) -> str:
        _ = self.rgy_per_frame()
        rg_per_frame = np.loadtxt(self.rgy_file, delimiter=",", skiprows=1)
        fig_analysis = f"rgy_{self.traj_file}"
        plot_name = self.path_registry.write_file_name(
            type=FileType.FIGURE, fig_analysis=fig_analysis, file_format="png"
        )
        print("plot_name: ", plot_name)
        plot_id = self.path_registry.get_fileid(
            file_name=plot_name, type=FileType.FIGURE
        )
        plot_path = f"{self.path_registry.ckpt_figures}/{plot_name}"
        plot_path = plot_path if plot_path.endswith(".png") else plot_path + ".png"
        print("plot_path", plot_path)
        plt.plot(rg_per_frame)
        plt.xlabel("Frame")
        plt.ylabel("Radius of Gyration (nm)")
        plt.title(f"{self.traj_file} - Radius of Gyration Over Time")

        plt.savefig(f"{plot_path}")
        self.path_registry.map_path(
            plot_id,
            plot_path,
            description=f"Plot of radii of gyration over time for {self.traj_file}",
        )
        plt.close()
        plt.clf()
        return "Plot saved as: " + f"{plot_name} with plot ID {plot_id}"


class RadiusofGyrationAverage(BaseTool):
    name = "RadiusofGyrationAverage"
    description = """This tool calculates the average radius of gyration
    for a trajectory. Give this tool BOTH the trajectory file ID and the
    topology file ID."""

    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file: str, top_file: str) -> str:
        """use the tool."""
        RGY = RadiusofGyration(self.path_registry)
        try:
            RGY._load_traj(top_file=top_file, traj_file=traj_file)
        except Exception as e:
            return f"Error loading traj: {e}"
        try:
            return "Succeeded. " + RGY.rgy_average()
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
    at each frame of a given trajectory.
    Give this tool BOTH the trajectory file ID and the
    topology file ID.
    The tool will save the radii of gyration to a csv file and
    map it to the registry."""

    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file: str, top_file: str) -> str:
        """use the tool."""
        RGY = RadiusofGyration(self.path_registry)
        try:
            RGY._load_traj(top_file=top_file, traj_file=traj_file)
        except Exception as e:
            return f"Error loading traj: {e}"
        try:
            return "Succeeded. " + RGY.rgy_per_frame()
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
    Give this tool BOTH the trajectory file ID and the
    topology file ID.
    The tool will save the plot to a png file and map it to the registry."""

    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file: str, top_file: str) -> str:
        """use the tool."""
        RGY = RadiusofGyration(self.path_registry)
        try:
            RGY._load_traj(top_file=top_file, traj_file=traj_file)
        except Exception as e:
            return f"Error loading traj: {e}"
        try:
            return "Succeeded. " + RGY.plot_rgy()
        except ValueError as e:
            return f"Failed. ValueError: {e}"
        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
