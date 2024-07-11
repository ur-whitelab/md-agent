import json
import os

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
from langchain.tools import BaseTool

from mdagent.utils import PathRegistry, load_single_traj


class HydrogenBondTool(BaseTool):
    name = "hydrogen_bond_tool"
    description = """Identifies hydrogen bonds using different methods;
    Baker-Hubbard and Wernet-Nilsson. Input a trajectory file ID and a method (either baker_hubbard or wernet_nilsson). If baker_hubbard is used, a frequency must be provided as a float. Optionally provide the topology file ID. Output is a file and plot of the hydrogen bonds found.

    """

    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(
        self,
        traj_file: str,
        top_file: str | None = None,
        method: str = "baker_hubbard",
        freq: str | None = "0.1",
    ) -> str:
        try:

            traj = load_single_traj(self.path_registry, top_file, traj_file)
            if not traj:
                return """Failed. Trajectory could not be loaded; unable to retrieve
                data needed to find hydrogen bonds. This may be due to missing files,
                corrupted files, or incorrect formatted file. Please check and try
                again."""

            # baker_hubbard is the default method if user doesnt specify which method

            if method == "wernet_nilsson":
                result = md.wernet_nilsson(
                    traj,
                    exclude_water=True,
                    periodic=True,
                    sidechain_only=False,
                )
            else:
                frequency = float(freq) if freq else 0.1
                result = md.baker_hubbard(
                    traj,
                    frequency,
                    exclude_water=True,
                    periodic=True,
                    sidechain_only=False,
                )

            # Count the number of hydrogen bonds for each frame
            hb_counts = np.array([len(frame) for frame in result])

            if self.path_registry is not None:
                self.save_results_to_file(
                    {"results": [list(item) for item in result]},
                    f"{method}_results.json",
                )
                plot_save_path_hist = self.path_registry.get_mapped_path(
                    f"{method}_histogram_plot.png",
                )
                plot_histogram(
                    hb_counts,
                    title=f"{method.capitalize()} Histogram",
                    save_path=plot_save_path_hist,
                )

                plot_save_path_time_series = self.path_registry.get_mapped_path(
                    f"{method}_time_series_plot.png",
                )
                plot_time_series(
                    hb_counts,
                    title=f"{method.capitalize()} Time Series",
                    save_path=plot_save_path_time_series,
                )
                return """Succeeded. Analysis completed, results saved to file and plot
                saved."""

            else:
                return """Failed. Path registry helps track
                file locations and it is not set up. Please make sure it is set up
                before running this tool."""

        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"


    def save_results_to_file(self, results: dict, file_name: str) -> None:
        with open(file_name, "w") as f:
            json.dump(results, f)


class KabschSander(BaseTool):
    name = "kabsch_sander"
    description = """Compute the hydrogen bond energy between each pair
    of residues in every frame."""

    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file: str, top_file: str | None = None) -> str:
        try:
            if not top_file:
                top_file = self.top_file(traj_file)
            traj = load_single_traj(self.path_registry, top_file, traj_file)
            if not traj:
                return """Failed. Trajectory could not be loaded; unable to retrieve
            data needed to find hydrogen bonds. This may be due to missing files,
            corrupted files, or incorrect formatted file. Please check and try
            again."""

            result = md.kabsch_sander(traj)

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

    async def _arun(self, traj_file, top_file=None):
        raise NotImplementedError("Async version not implemented")

    def top_file(self, traj_file: str) -> str:
        top_file = os.path.join(os.path.dirname(traj_file), "topology.pdb")
        return top_file

    def save_results_to_file(self, results: dict, file_name: str) -> None:
        with open(file_name, "w") as f:
            json.dump(results, f)


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

   plt.savefig(save_path)
    plt.close()


