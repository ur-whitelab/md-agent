import json
import os

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
from langchain.tools import BaseTool

from mdagent.utils import FileType, PathRegistry, load_single_traj


def compute_baker_hubbard(traj, freq=0.1):
    """
    Computes hydrogen bonds using the Baker-Hubbard method.

    Args:
        traj: The trajectory data.
        freq: The frequency cutoff.

    Returns:
        The hydrogen bonds found using the Baker-Hubbard method.
    """
    try:
        frequency = float(freq)
    except ValueError:
        raise ValueError("Frequency must be a float.")

    return md.baker_hubbard(
        traj,
        frequency,
        exclude_water=True,
        periodic=True,
        sidechain_only=False,
    )


def compute_wernet_nilsson(traj):
    return md.wernet_nilsson(
        traj,
        exclude_water=True,
        periodic=True,
        sidechain_only=False,
    )


def save_hb_results(results: dict, method: str, path_registry: PathRegistry) -> str:
    file_name = path_registry.write_file_name(
        FileType.RECORD,
        record_type=f"{method}_results",
    )
    file_id = path_registry.get_fileid(file_name, FileType.RECORD)
    file_path = f"{path_registry.ckpt_records}/{method}_results.json"
    with open(file_path, "w") as f:
        json.dump(results, f)
    path_registry.map_path(
        file_id,
        file_name,
        description=f"Hydrogen bond results for {method}",
    )
    return file_id


def plot_and_save_hb_plot(
    data: np.ndarray,
    title: str,
    plot_type: str,
    method: str,
    path_registry: PathRegistry,
    ylabel: str = "Value",  # Added ylabel parameter with default value
) -> str:
    """
    Plots the data and saves the plot to a file.

    Args:
    data: The data to plot.
    title: The title of the plot.
    plot_type: The type of plot ('histogram' or 'time_series').
    method: The method name used for file naming.
    path_registry: The path registry to save the file to.

    Returns:
    The file ID of the saved plot.
    """

    plt.figure(figsize=(10, 6))
    if plot_type == "histogram":
        plt.hist(data, bins=10, edgecolor="black")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
    elif plot_type == "time_series":
        plt.plot(data, label="Hydrogen Bonds")
        plt.xlabel("Time (frames)")
        plt.ylabel("Value")
    plt.title(title)
    plt.grid(True)

    file_name = path_registry.write_file_name(
        FileType.FIGURE,
        file_format="png",
    )
    file_id = path_registry.get_fileid(file_name, FileType.FIGURE)

    file_path = f"{path_registry.ckpt_figures}/{method}_{plot_type}.png"

    plt.savefig(file_path, format="png", dpi=300, bbox_inches="tight")

    plt.close()

    path_registry.map_path(
        file_id,
        file_name,
        description=(f"{title} for {method}"),
    )
    return file_id


class HydrogenBondTool(BaseTool):
    name = "hydrogen_bond_tool"
    description = (
        "Identifies hydrogen bonds using different methods: Baker-Hubbard or "
        " Wernet-Nilsson, and plots the results from the provided trajectory data."
        "\nInputs: \n"
        "\t(str) File ID for the trajectory file. \n"
        "\t(str, optional) File ID for the topology file. \n"
        "\t(str) Method to use for identification ('baker_hubbard' or "
        "'wernet_nilsson'). \n"
        "\t(float, optional) Frequency for the Baker-Hubbard method (default: 0.1). \n"
        "\nOutputs: \n"
        "\t(str) Result of the analysis indicating success or failure, along with file"
        "IDs for results and plots."
    )

    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(
        self,
        traj_file: str,
        top_file: str | None = None,
        method: str = "baker_hubbard",
        freq: str = "0.1",
    ) -> str:
        if self.path_registry is None:
            raise ValueError("Path registry is not set.")
        try:
            traj = load_single_traj(self.path_registry, top_file, traj_file)
            if not traj:
                return """Failed. Trajectory could not be loaded; unable to retrieve
                data needed to find hydrogen bonds. This may be due to missing files,
                corrupted files, or incorrect formatted file. Please check and try
                again."""

            # Call the appropriate helper function based on the method
            if method == "wernet_nilsson":
                result = compute_wernet_nilsson(traj)
            else:
                result = compute_baker_hubbard(traj, freq)

            # Count the number of hydrogen bonds for each frame
            hb_counts = np.array([len(frame) for frame in result])

            result_file_id = save_hb_results(
                {"results": [list(item) for item in result]},
                method,
                self.path_registry,
            )

            plot_hist_file_id = plot_and_save_hb_plot(
                hb_counts,
                title=f"{method.capitalize()} Histogram",
                plot_type="histogram",
                method=method,
                path_registry=self.path_registry,
            )

            plot_time_series_file_id = plot_and_save_hb_plot(
                hb_counts,
                title=f"{method.capitalize()} Time Series",
                plot_type="time_series",
                method=method,
                path_registry=self.path_registry,
                ylabel="Bond Energy",
            )
            return (
                "Succeeded. Analysis completed, results saved to file and plot"
                "saved. "
                f"Results file: {result_file_id}, "
                f"Histogram plot: {plot_hist_file_id}, "
                f"Time series plot: {plot_time_series_file_id}"
            )

        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"


class KabschSander(BaseTool):
    name = "kabsch_sander"
    description: str = (
        "Compute the hydrogen bond energy between each pair of residues"
        "in every frame of the trajectory."
    )

    "\n Parameters: \n"
    "\t(str) traj_file: The file ID of the trajectory file containing "
    "molecular dynamics (MD) data.\n"
    "\t(str, optional)top_file: The optional topology file ID"
    "providing molecular structure data. Default is None. \n"

    "\n Returns:\n"
    "\t(str): A message indicating whether the analysis was successful or failed."

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
                plot_time_series_file_id = plot_and_save_hb_plot(
                    result[1],  # assuming result[1] contains the bond energies
                    title="Kabsch-Sander Time Series",
                    plot_type="time_series",
                    method="kabsch_sander",
                    path_registry=self.path_registry,
                    ylabel="Bond Energy",
                )
                return (
                    "Succeeded. Kabsch-Sander analysis completed, results saved to "
                    f"file and plot saved. Plot file: {plot_time_series_file_id}"
                )
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
        if self.path_registry:
            file_id = self.path_registry.get_fileid(file_name, FileType.RECORD)
            self.path_registry.map_path(
                file_id,
                file_name,
                description=f"Results saved to {file_name}",
            )
