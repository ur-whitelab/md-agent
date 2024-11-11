import itertools
from typing import List, Optional

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


def save_hb_results(result, method: str, path_registry: PathRegistry) -> str:
    if method == "kabsch_sander":
        data = []
        for frame_idx, sparse_matrix in enumerate(result):
            coo_matrix = sparse_matrix.tocoo()
            # converts to COO format for easy iteration
            for i, j, energy in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
                data.append([frame_idx, i, j, energy])
        df = pd.DataFrame(
            data,
            columns=["frame", "residue_i", "residue_j", "energy"],
        )
    elif method == "wernet_nilsson":
        data = []
        for frame_index, frame in enumerate(result):
            for bond in frame:
                donor, hydrogen, acceptor = bond
                data.append([frame_index, donor, hydrogen, acceptor])

        df = pd.DataFrame(
            data,
            columns=["donor atom", "h atom", "acceptor atom"],
        )
    else:
        df = pd.DataFrame(
            result,
            columns=["donor atom", "h atom", "acceptor atom"],
        )

    file_name = path_registry.write_file_name(
        FileType.RECORD,
        record_type=f"{method}_results",
        file_format="csv",
    )
    file_id = path_registry.get_fileid(file_name, FileType.RECORD)
    file_path = f"{path_registry.ckpt_records}/{file_name}"
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
    ylabel: str = "Count",
    annotations: Optional[List[str]] = None,
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
        color = itertools.cycle(["r", "b", "gold"])

        if annotations is not None:
            for i in most_frequent_indices:
                plt.hist(
                    data[:, i],
                    color=next(color),
                    label=annotations[i],
                    alpha=0.5,
                )
        else:
            for i in most_frequent_indices:
                plt.hist(data[:, i], color=next(color), alpha=0.5)

        plt.xlabel("Donor-acceptor distance [nm]")
        plt.ylabel("Frequency")
    elif plot_type == "time_series":
        plt.plot(data, label="Hydrogen Bonds")
        plt.xlabel("Time (frames)")
        plt.ylabel(ylabel)

    plt.title(title)
    plt.grid(True)
    plt.legend()

    file_id = save_plot(
        path_registry,
        fig_analysis=f"{method}_{plot_type}",
        description=f"{title} for {method}",
    )
    plt.close()

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

    def __init__(self, path_registry):
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
            return "PathRegistry is not set"

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
            else:  # this is the baker-hubbard method
                result = compute_baker_hubbard(traj, freq)

            result_file_id = save_hb_results(
                result,
                method,
                self.path_registry,
            )

            if method == "wernet_nilsson":
                # Count the number of hydrogen bonds for each frame
                hb_counts = np.array([len(frame) for frame in result])
                # da_distances = hb_counts

                plot_file_id = plot_and_save_hb_plot(
                    hb_counts,
                    title=f"{method.capitalize()} Time Series",
                    plot_type="time_series",
                    method=method,
                    path_registry=self.path_registry,
                    ylabel="Count",
                )

            else:  # this is the baker-hubbard method
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
                    ]
                )

                plot_file_id = plot_and_save_hb_plot(
                    da_distances,
                    title=f"{method.capitalize()} Histogram - Top 3 HBonds",
                    plot_type="histogram",
                    method=method,
                    path_registry=self.path_registry,
                    annotations=annotations,
                )

            return (
                "Succeeded. Analysis completed, results saved to file and plot "
                "saved. "
                f"Results file: {result_file_id}, "
                f"Histogram or Time series plot: {plot_file_id}, "
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

        if self.path_registry is None:
            return "PathRegistry is not set"
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

            result_file_id = save_hb_results(
                result,
                method="kabsch_sander",
                path_registry=self.path_registry,
            )

            total_energies = [matrix.sum() for matrix in result]
            plot_time_series_file_id = plot_and_save_hb_plot(
                total_energies,
                title="Kabsch-Sander Time Series",
                plot_type="time_series",
                method="kabsch_sander",
                path_registry=self.path_registry,
                ylabel="Total HBond Energy (kcal/mol)",
            )

            return (
                "Succeeded. Kabsch-Sander analysis completed, results saved to file "
                "and plot saved. "
                f"Results file:{result_file_id}, "
                f"Plot file: {plot_time_series_file_id}, "
            )

        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"

    async def _arun(self, traj_file, top_file=None):
        raise NotImplementedError("Async version not implemented")
