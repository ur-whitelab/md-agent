import os
from typing import Optional, Type

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
        print("Warning: no trajectory file provided.")
        return md.load(top_path)

    if traj_fileid not in all_fileids:
        raise ValueError("Trajectory File ID not found in path registry")

    traj_path = path_registry.get_mapped_path(traj_fileid)
    return md.load(traj_path, top=top_path)


def save_to_csv(path_registry, data, file_id, description=None):
    file_path = f"{path_registry.ckpt_figures}/{file_id}.csv"
    i = 0
    while os.path.exists(file_path):
        i += 1
        file_path = f"{path_registry.ckpt_figures}/{file_id}_{i}.csv"
    np.savetxt(file_path, data, delimiter=",")
    path_registry.map_path(file_id, file_path, description=description)
    return file_path


def save_plot(path_registry, fig_analysis, description=None):
    fig_name = path_registry.write_file_name(
        type=FileType.FIGURE,
        fig_analysis=fig_analysis,
        file_format="png",
    )
    fig_id = path_registry.get_fileid(file_name=fig_name, type=FileType.FIGURE)
    fig_path = f"{path_registry.ckpt_figures}/{fig_name}"
    plt.savefig(fig_path)
    path_registry.map_path(fig_id, fig_path, description=description)
    return fig_id, fig_path


def rmsd(
    path_registry, top_id, traj_id=None, ref_top_id=None, ref_traj_id=None, select="all"
):
    print("Calculating RMSD...")
    traj = load_traj(path_registry, top_id, traj_id)
    if ref_top_id is None:
        ref_traj = traj
    else:
        ref_traj = load_traj(path_registry, ref_top_id, ref_traj_id)
    idx = traj.topology.select(select)
    traj.center_coordinates()
    rmsd = md.rmsd(traj, ref_traj, precentered=True)
    rmsd_select = md.rmsd(traj, ref_traj, atom_indices=idx, precentered=True)

    if rmsd.shape[0] == 1:  # if it's single value
        return f"RMSD calculated. {rmsd[0]} nm"
    data_id = top_id.replace("top_", "")
    analysis = f"rmsd_{data_id}"
    csv_path = save_to_csv(
        path_registry, rmsd, analysis, description=f"RMSD for {data_id}"
    )

    # plot rmsd
    fig, ax = plt.subplots()
    ax.plot(rmsd, label="protein")
    ax.plot(rmsd_select, label=select)
    ax.legend()
    ax.set(
        **{
            "xlabel": "time",
            "ylabel": "RMSD / nm",
        }
    )
    fig_id, fig_path = save_plot(path_registry, analysis, f"RMSD plot for {data_id}")
    plt.close()
    msg = (
        f"RMSD calculated and saved as {csv_path} with file ID rmsd_{data_id}"
        f"Plot saved to {fig_path} with plot ID {fig_id}. "
    )
    return msg


def rmsf(
    path_registry, top_id, traj_id=None, ref_top_id=None, ref_traj_id=None, select="all"
):
    print("Calculating RMSF...")
    traj = load_traj(path_registry, top_id, traj_id)
    if ref_top_id is None:
        ref_traj = traj
    else:
        ref_traj = load_traj(path_registry, ref_top_id, ref_traj_id)
    idx = traj.topology.select(select)
    traj.center_coordinates()
    rmsf = md.rmsf(traj, ref_traj, precentered=True)
    rmsf_select = rmsf[idx]
    data_id = top_id.replace("top_", "")
    analysis = f"rmsf_{data_id}"
    csv_path = save_to_csv(
        path_registry, rmsf_select, analysis, description=f"RMSF for {data_id}"
    )

    # plot rmsf
    if select == "all" or select == "backbone":
        select = "sidechain"
    select_idx = traj.topology.select(select)
    backbone_idx = traj.topology.select("backbone")

    fig, ax = plt.subplots()
    ax.bar(
        select_idx,
        rmsf[select_idx],
        width=1,
        edgecolor="k",
        linewidth=0.2,
        label="sidechain",
    )
    ax.bar(
        backbone_idx,
        rmsf[backbone_idx],
        width=1,
        edgecolor="k",
        linewidth=0.2,
        label="backbone",
    )
    ax.legend()
    ax.set(
        **{
            "xlabel": "atom index",
            "ylabel": "RMSF / nm",
        }
    )
    fig_id, fig_path = save_plot(path_registry, analysis, f"RMSF plot for {data_id}")
    plt.close()
    msg = (
        f"RMSF calculated and saved as {csv_path} with file ID rmsf_{data_id}"
        f"Plot saved to {fig_path} with plot ID {fig_id}. "
    )
    return msg


def lprmsd(
    path_registry, top_id, traj_id=None, ref_top_id=None, ref_traj_id=None, select="all"
):
    print("Calculating LP-RMSD...")
    traj = load_traj(path_registry, top_id, traj_id)
    if ref_top_id is None:
        ref_traj = traj
    else:
        ref_traj = load_traj(path_registry, ref_top_id, ref_traj_id)
    idx = traj.topology.select(select)
    lprmsd = md.rmsd(traj, ref_traj, atom_indices=idx, precentered=True)
    data_id = top_id.replace("top_", "")
    csv_path = save_to_csv(
        path_registry, lprmsd, f"lprmsd_{data_id}", description=f"LP-RMSD for {data_id}"
    )
    return f"LP-RMSD calculated and saved as {csv_path} with file ID lprmsd_{data_id}"


class RMSDInputSchema(BaseModel):
    top_id: str = Field(None, description="File ID for the topology file.")
    traj_id: Optional[str] = Field(None, description="File ID for the trajectory file.")
    ref_top_id: Optional[str] = Field(
        None, description="File ID for the topology file as reference"
    )
    ref_traj_id: Optional[str] = Field(
        None, description="File ID for the trajectory file as reference."
    )
    select: Optional[str] = Field(
        "all",
        description=(
            "atom selection following MDTraj syntax for topology.select. "
            "Examples are 'all', 'backbone', 'sidechain'"
        ),
    )


class ComputeRMSD(BaseTool):
    name: str = "ComputeRMSD"
    description: str = (
        "Compute root mean square deviation (RMSD) of all "
        "conformations in target to a reference conformation"
    )
    args_schema: Type[BaseModel] = RMSDInputSchema
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry] = None):
        super().__init__()
        self.path_registry = path_registry

    def _run(
        self, top_id, traj_id=None, ref_top_id=None, ref_traj_id=None, select=None
    ):
        try:
            msg = rmsd(
                self.path_registry, top_id, traj_id, ref_top_id, ref_traj_id, select
            )
            return f"Succeeded. {msg}"
        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"


class ComputeRMSF(BaseTool):
    name: str = "ComputeRMSF"
    description: str = (
        "Compute root mean square fluctuation (RMSF) of all "
        "conformations in target to a reference conformation"
    )
    args_schema: Type[BaseModel] = RMSDInputSchema
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry] = None):
        super().__init__()
        self.path_registry = path_registry

    def _run(
        self, top_id, traj_id=None, ref_top_id=None, ref_traj_id=None, select=None
    ):
        try:
            msg = rmsf(
                self.path_registry, top_id, traj_id, ref_top_id, ref_traj_id, select
            )
            return f"Succeeded. {msg}"
        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"


class ComputeLPRMSD(BaseTool):
    name: str = "ComputeLP-RMSD"
    description: str = (
        "Compute Linear-Programming Root-Mean-Squared Deviation "
        "(LP-RMSD) of all conformations in target to a reference "
        "conformation. The LP-RMSD is the minimum RMSD between "
        "two sets of points, minimizing over both the rotational/"
        "translational degrees of freedom AND the label "
        "correspondences between points in the target and "
        "reference conformations."
    )
    args_schema: Type[BaseModel] = RMSDInputSchema
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry] = None):
        super().__init__()
        self.path_registry = path_registry

    def _run(
        self, top_id, traj_id=None, ref_top_id=None, ref_traj_id=None, select=None
    ):
        try:
            msg = lprmsd(
                self.path_registry, top_id, traj_id, ref_top_id, ref_traj_id, select
            )
            return f"Succeeded. {msg}"
        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"
