import itertools
import json
import os

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import pandas as pd
from langchain.tools import BaseTool

from mdagent.utils import FileType, PathRegistry, load_single_traj, save_plot


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


def save_hb_results(
    result: dict,
    method: str,
    path_registry: PathRegistry,
) -> str:
    if method == "wernet_nilsson":
        data = []
        for frame_index, frame in enumerate(result):
            for bond in frame:
                donor, hydrogen, acceptor = bond
                data.append([frame_index, donor, hydrogen, acceptor])

        df = pd.DataFrame(
            data,
            columns=["frame", "donor atom", "h atom", "acceptor atom"],
        )
    else:
        data = []
        for frame_index, frame in enumerate(result):
            for bond in frame:
                donor, hydrogen, acceptor = bond
                data.append([frame_index, donor, hydrogen, acceptor])

        df = pd.DataFrame(
            data,
            columns=["donor atom", "h atom", "acceptor atom"],
        )

    file_name = path_registry.write_file_name(
        FileType.RECORD,
        record_type=f"{method}_results",
    )
    file_id = path_registry.get_fileid(file_name, FileType.RECORD)
    file_path = f"{path_registry.ckpt_records}/{file_name}.csv"
    df.to_csv(file_path, index=False)
    path_registry.map_path(
        file_id,
        file_path,
        description=f"Hydrogen bond results for {method}",
    )
    return file_id


def plot_and_save_hb_plot(
    data: np.ndarray,
    title: str,
    plot_type: str,
    method: str,
    path_registry: PathRegistry,
    annotations: list | None = None,
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
        hbond_frequencies = np.mean(data < 0.25, axis=0)
        most_frequent_indices = np.argsort(-hbond_frequencies)[:3]
        color = itertools.cycle(["r", "g", "blue"])
        for i in most_frequent_indices:
            if annotations:
                label = annotations[i]
            else:
                label = f"Hydrogen Bond {i}"

            plt.hist(
                data[:, i],
                color=next(color),
                label=label,
                alpha=0.5,
                edgecolor="black",
            )
        plt.xlabel("Donor-acceptor distance [nm]")
        plt.ylabel("Frequency")
    elif plot_type == "time_series":
        plt.plot(data, label="Hydrogen Bonds")
        plt.xlabel("Frame Number")
        plt.ylabel("Count")

    plt.title(title)
    plt.grid(True)
    plt.legend()

    file_id = save_plot(
        path_registry,
        fig_analysis=f"{method}_{plot_type}",
        description=f"{title} for {method}",
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
        try:
            print("Loading trajectory...")
            traj = load_single_traj(
                self.path_registry,
                top_file,
                traj_file,
            )
            if not traj:
                return """Failed. Trajectory could not be loaded; unable to retrieve
                data needed to find hydrogen bonds. This may be due to missing files,
                corrupted files, or incorrect formatted file. Please check and try
                again."""

            if method == "wernet_nilsson":
                result = compute_wernet_nilsson(traj)
            else:
                result = compute_baker_hubbard(traj, freq)

            if self.path_registry is None:
                raise ValueError("PathRegistry is not set")

            result_file_id = save_hb_results(
                {i: res for i, res in enumerate(result)},
                method,
                self.path_registry,
            )
            if method == "wernet_nilsson":
                hb_counts = np.array([len(frame) for frame in result])
                if self.path_registry is None:
                    raise ValueError("PathRegistry is not set")

                plot_file_id = plot_and_save_hb_plot(
                    hb_counts,
                    title=f"{method.capitalize()} Time Series",
                    plot_type="Time Series",
                    method=method,
                    path_registry=self.path_registry,
                )
            else:
                # compute distance between H bonds
                da_distances = md.compute_distances(
                    traj,
                    result[:, [0, 2]],
                    periodic=False,
                )

                annotations = np.array(
                    [
                        "%s -- %s"
                        % (traj.topology.atom(hbond[0]), traj.topology.atom(hbond[2]))
                        for hbond in result
                    ],
                )
                if self.path_registry is None:
                    raise ValueError("PathRegistry is not set")

                plot_file_id = plot_and_save_hb_plot(
                    da_distances,
                    title=f"{method.capitalize()} Histogram - Top 3 HBonds",
                    plot_type="histogram",
                    method=method,
                    path_registry=self.path_registry,
                    annotations=annotations,
                )

            return (
                "Succeeded. Analysis completed, results saved to file and plot"
                "saved. "
                f"Results file: {result_file_id}, "
                f"Histogram plot or time series plot: {plot_file_id}, "
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

            if self.path_registry is None:
                raise ValueError("PathRegistry is not set")

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

                if self.path_registry is None:
                    raise ValueError("PathRegistry is not set")

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
