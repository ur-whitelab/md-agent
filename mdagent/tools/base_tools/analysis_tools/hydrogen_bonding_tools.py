import json
import os

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
from langchain.tools import BaseTool

from mdagent.utils import PathRegistry, load_single_traj


class HydrogenBondTool(BaseTool):
    name = "hydrogen_bond_tool"
    description = """Base class for hydrogen bond analysis tools."""

    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file: str, top_file: str | None = None) -> str:
        raise NotImplementedError("Subclasses should implement this method.")

    async def _arun(self, traj_file: str, top_file: str | None = None) -> str:
        raise NotImplementedError("Async version not implemented")

    def save_results_to_file(self, results: dict, file_name: str) -> None:
        with open(file_name, "w") as f:
            json.dump(results, f)


class BakerHubbard(HydrogenBondTool):
    name = "baker_hubbard"
    description = """Identify hydrogen bonds that are present in at least 10% of each
    frame (freq=0.1). Provides a list of tuples with each tuple containing three
    integers representing the indices of atoms (donor, hydrogen, acceptor) involved in
    the hydrogen bonding."""

    exclude_water: bool
    periodic: bool
    sidechain_only: bool

    def __init__(
        self,
        path_registry: PathRegistry,
        exclude_water=True,
        periodic=True,
        sidechain_only=False,
    ):
        super().__init__(path_registry)
        self.exclude_water = exclude_water
        self.periodic = periodic
        self.sidechain_only = sidechain_only

    def _run(
        self,
        traj_file: str,
        top_file: str | None = None,
        freq: float = 0.1,
    ) -> str:
        try:
            if not top_file:
                top_file = self.top_file(traj_file)

            traj = load_single_traj(self.path_registry, top_file, traj_file)
            if not traj:
                return """Failed. Trajectory could not be loaded; unable to retrieve
                data needed to find hydrogen bonds. This may be due missing files,
                corrupted files, or incorrect formatted file. Please check and try
                again."""

            result = md.baker_hubbard(
                traj,
                freq,
                exclude_water=self.exclude_water,
                periodic=self.periodic,
                sidechain_only=self.sidechain_only,
            )

            # Count the number of hydrogen bonds for each frame
            hb_counts = np.array([len(frame) for frame in result])

            # Check to see if path_registry is not None

            if self.path_registry is not None:
                self.save_results_to_file(
                    {"results": result.tolist()},
                    "baker_hubbard_results.json",
                )
                plot_save_path_hist = self.path_registry.get_mapped_path(
                    "baker_hubbard_histogram_plot.png",
                )
                plot_histogram(
                    hb_counts,
                    title="Baker-Hubbard Histogram",
                    save_path=plot_save_path_hist,
                )

                plot_save_path_time_series = self.path_registry.get_mapped_path(
                    "baker_hubbard_time_series_plot.png",
                )
                plot_time_series(
                    hb_counts,
                    title="Baker-Hubbard Time Series",
                    save_path=plot_save_path_time_series,
                )
                return """Succeeded. Baker-Hubbard analysis completed, results saved to
                file and plot saved."""
            else:
                return """Failed. Path registry helps track
            file locations and it is not set up. Please make sure it is set up
            before running this tool."""

        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"

    def top_file(self, traj_file: str) -> str:
        top_file = os.path.join(os.path.dirname(traj_file), "topology.pdb")
        return top_file


class KabschSander(HydrogenBondTool):
    name = "kabsch_sander"
    description = """Calculate the energy of hydrogen bonds between pairs of
    residues in each frame of the simulation. It shows which residues are
    forming hydrogen bonds and the energy of these bonds for
      each frame."."""

    def _run(self, traj_file: str, top_file: str | None = None) -> str:
        try:
            if not top_file:
                top_file = self.top_file(traj_file)
            traj = load_single_traj(self.path_registry, top_file, traj_file)
            if not traj:
                return """Failed. Trajectory could not be loaded; unable to access
            data required to calculate hydrogen bond energies. This could be due to
            missing files, corrupted files, or incorrect formatted file. Please check
            and try again."""

            result = md.kabsch_sander(traj)

            # Check to see if path_registry is not None

            if self.path_registry is not None:
                result_dict = {
                    "indices": [
                        list(pair) for pair in result[0]
                    ],  # Convert each tuple to a list
                    "energies": result[1].tolist()
                    if hasattr(result[1], "tolist")
                    else list(result[1]),
                }
                self.save_results_to_file(result_dict, "kabsch_sander_results.json")
                plot_save_path_time_series = self.path_registry.get_mapped_path(
                    "kabsch_sander_time_series_plot.png",
                )
                plot_time_series(
                    result[1],  # assuming result[1] contains the bond energies
                    title="Kabsch-Sander Time Series",
                    ylabel="Bond Energy",
                    save_path=plot_save_path_time_series,
                )
                return """Succeeded. Kabsch-Sander analysis completed, results saved
                to file and plot saved."""
            else:
                return """Failed. Path registry helps track
                file locations and it is not set up. Please make sure it is set up
                before running this tool."""

        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"

    def top_file(self, traj_file: str) -> str:
        top_file = os.path.join(os.path.dirname(traj_file), "topology.pdb")
        return top_file


class WernetNilsson(HydrogenBondTool):
    name = "wernet_nilsson"
    description = """Identifies hydrogen bonds without a frequency parameter. Provides
    a list of tuples with indices of donor, hydrogen, and acceptor atoms. Prefer this
    tool over BakerHubbard unless explicitly requested."""

    exclude_water: bool
    periodic: bool
    sidechain_only: bool

    def __init__(
        self,
        path_registry: PathRegistry,
        exclude_water=True,
        periodic=True,
        sidechain_only=False,
    ):
        super().__init__(path_registry)
        self.exclude_water = exclude_water
        self.periodic = periodic
        self.sidechain_only = sidechain_only

    def _run(self, traj_file: str, top_file: str | None = None) -> str:
        try:
            if not top_file:
                top_file = self.top_file(traj_file)
            traj = load_single_traj(self.path_registry, top_file, traj_file)
            if not traj:
                return """Failed. Trajectory could not be loaded' unable to retrieve
                data needed to find hydrogen bonds. This may be due missing files,
                corrupted files, or incorrect formatted file. Please check and try
                again"""

            result = md.wernet_nilsson(
                traj,
                exclude_water=self.exclude_water,
                periodic=self.periodic,
                sidechain_only=self.sidechain_only,
            )
            # Check to see if path_registry is not None

            if self.path_registry is not None:
                self.save_results_to_file(
                    {"results": result.tolist()},
                    "wernet_nilsson_results.json",
                )
                plot_save_path_hist = self.path_registry.get_mapped_path(
                    "wernet_nilsson_histogram_plot.png",
                )
                plot_histogram(
                    [len(r) for r in result],
                    title="Wernet-Nilsson Histogram",
                    save_path=plot_save_path_hist,
                )

                plot_save_path_time_series = self.path_registry.get_mapped_path(
                    "wernet_nilsson_time_series_plot.png",
                )
                plot_time_series(
                    [len(r) for r in result],
                    title="Wernet-Nilsson Time Series",
                    save_path=plot_save_path_time_series,
                )
                return """Succeeded. Wernet-Nilsson analysis completed, results
                saved to file and plot saved."""
            else:
                return """Failed. Path registry helps track
                file locations and it is not set up. Please make sure it is set up
                before running this tool."""

        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"

    def top_file(self, traj_file: str) -> str:
        top_file = os.path.join(os.path.dirname(traj_file), "topology.pdb")
        return top_file


# Test instantiation and running of each tool
if __name__ == "__main__":
    path_registry = PathRegistry.get_instance("path_to_ckpt")
    traj_file = "path_to_traj_file.dcd"
    top_file = "path_to_topology_file.pdb"

    user_input = (
        input(
            """Please specify the hydrogen bond tool to use
            (baker_hubbard/wernet_nilsson): """
        )
        .strip()
        .lower()
    )

    if "baker_hubbard" in user_input:
        tool = BakerHubbard(path_registry)
        print(f"Running {tool.name}...")
        result = tool._run(traj_file, top_file)
        print(result)
    else:
        tool = WernetNilsson(path_registry)
        print(f"Running {tool.name}...")
        result = tool._run(traj_file, top_file)
        print(result)

    # Always run KabschSander regardless of user input
    tool = KabschSander(path_registry)
    print(f"Running {tool.name}...")
    result = tool._run(traj_file, top_file)
    print(result)

# Helper functions for plotting


def plot_time_series(data, title="Time Series Plot", ylabel="Value", save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(data, label="Hydrogen Bonds")
    plt.xlabel("Time (frames)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_histogram(data, bins=10, title="Histogram", xlabel="Value", save_path=None):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, edgecolor="black")
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


# Example usage of the plotting functions
if __name__ == "__main__":
    # Example data for plotting
    example_data = np.random.randn(100)

    # Plot time series
    plot_time_series(example_data, title="Example Time Series Plot")

    # Plot histogram
    plot_histogram(example_data, title="Example Histogram")
