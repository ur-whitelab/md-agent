from typing import Optional

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import pandas as pd
import seaborn as sns
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from mdagent.utils import FileType, PathRegistry


class PCA_analysis:
    def __init__(
        self,
        path_registry,
        pc_percentage,
        top_path,
        traj_path,
        sim_id,
        selection="backbone",
    ):
        self.path_registry = path_registry
        self.u = mda.Universe(top_path, traj_path)
        self.sim_id = sim_id
        if pc_percentage > 1:
            pc_percentage /= 100
        self.pc_percentage = pc_percentage
        self.selection = self.u.select_atoms(selection)

    def _align_trajectory(self):
        mda.analysis.align.AlignTraj(
            self.u, self.u, select="backbone", in_memory=True
        ).run()

    def get_pc(self, universe, selection="backbone"):
        self._align_trajectory()
        self.pc = mda.analysis.pca.PCA(
            universe,
            select=selection,
            align=True,
            mean=None,
            n_components=None,
            verbose=True,
        ).run()

    def _get_number_pcs(self):
        self.n_pcs = np.where(self.pc.results.cumulated_variance > self.pc_percentage)[
            0
        ][0]
        if self.n_pcs > 3:
            self.n_pcs = 3

        return self.n_pcs

    def make_transformation(
        self,
    ):
        self.transformed = self.pc.transform(self.backbone, n_components=self.n_pcs)

    def _make_df(self):
        self.pc_df = pd.DataFrame(
            self.transformed, columns=["PC{}".format(i + 1) for i in range(self.n_pcs)]
        )
        self.pc_df["Time (ps)"] = self.pc_df.index * self.u.trajectory.dt

    def make_scree_plot(self):
        cumulative_variance = self.pc.cumulated_variance
        plt.plot(1 - cumulative_variance)
        # Calculate the index where cumulative variance exceeds or meets 95%
        threshold = self.pc_percentage
        threshold_index = next(
            x[0] for x in enumerate(cumulative_variance) if x[1] >= threshold
        )

        # Add a horizontal dashed line at 95% threshold
        plt.axhline(
            y=threshold,
            color="r",
            linestyle="--",
            label=f"95% Contribution at PC{threshold_index+1}",
        )

        plt.xlabel("Principal component")
        plt.ylabel("EigenValue Proportion")
        plt.legend()

        desc = f"Scree Plot of the eigenvalues of simulation {self.sim_id}"
        file_name = self.path_registry.write_file_name(
            FileType.FIGURE,
            fig_analysis="scree_plot",
            file_format="png",
            Sim_id=self.sim_id,
        )
        plot_id = self.path_registry.get_fileid(file_name, FileType.FIGURE)
        path = self.path_registry.ckpt_dir + "/figures/"
        plt.savefig(path + file_name)  # Save the figure before plt.show()
        self.path_registry.map_path(plot_id, path + file_name, description=desc)

    def make_pc_plots(self):
        g = sns.PairGrid(
            self.pc_df,
            hue="Time (ps)",
            palette=sns.color_palette("Oranges_d", n_colors=len(self.pc_df)),
        )
        g.map(plt.scatter, marker=".")
        desc = f"PCA Plot comparing the top {self.n_pcs} principal components"
        file_name = self.path_registry.write_file_name(
            FileType.FIGURE, fig_analysis="pca", file_format="png", Sim_id=self.sim_id
        )
        plot_id = self.path_registry.get_fileid(file_name, FileType.FIGURE)
        path = self.path_registry.ckpt_dir + "/figures/"
        plt.savefig(path + file_name)  # Save the figure before plt.show()
        self.path_registry.map_path(plot_id, path + file_name, description=desc)


class PCASchema(BaseModel):
    trajectory_id: str = Field("Trajectory File ID of the simulation to be analyzed")
    topology_id: str = Field("Topology File ID ot the simulation to be analyzed")
    pc_percentage: Optional[str] = Field(95, "Max cumulative percentage of components")


class PCATool(BaseTool):
    name = "PCATool"
    description = "Calculate the Principal Analysis Components of a " "MD trajectory"
    args_schema = PCASchema
    path_registry: Optional[PathRegistry]

    def _run(self, input):
        PCA_container = PCA_analysis(**input)
        PCA_container.get_pc()
        PCA_container.get_numbes_pcs()
        PCA_container.make_transformation()

        return "all done!"
