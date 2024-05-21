from typing import Optional

import matplotlib.pyplot as plt
import MDAnalysis as mda
import MDAnalysis.analysis.pca as pca
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
        self.atom_selection = self.u.select_atoms(selection)
        self.selection = selection
        self.pc = None
        self.n_pcs = None

    def _align_trajectory(self):
        try:
            mda.analysis.align.AlignTraj(
                self.u, self.u, select=self.selection, in_memory=True
            ).run()
            return "Trajectory aligned to the first frame. "
        except Exception as e:
            print(f"Error aligning trajectory: {str(e)}")
            return "Trajectory not aligned. Results may not be trustful"

    def get_pc(self):
        align_message = self._align_trajectory()
        self.pc = pca.PCA(
            self.u,
            select=self.selection,
            align=True,
            mean=None,
            n_components=None,
            verbose=True,
        ).run()
        return "PCA done" + align_message

    def _get_number_pcs(self):
        self.n_pcs = np.where(self.pc.results.cumulated_variance > self.pc_percentage)[
            0
        ][0]
        if self.n_pcs > 3:
            self.n_pcs = 3
        return self.n_pcs

    def _make_transformation(self):
        self.transformed = self.pc.transform(
            self.atom_selection, n_components=self.n_pcs
        )

    def _make_df(self):
        self._get_number_pcs()
        self._make_transformation()
        self.pc_df = pd.DataFrame(
            self.transformed, columns=["PC{}".format(i + 1) for i in range(self.n_pcs)]
        )
        self.pc_df["Time (ps)"] = self.pc_df.index * self.u.trajectory.dt

    def make_scree_plot(self):
        extra_mess = ""
        if not self.pc:
            pc_mess = self.get_pc()
            if "not aligned" in pc_mess:
                extra_mess += pc_mess

        cumulative_variance = self.pc.results.cumulated_variance
        print(cumulative_variance)
        plt.plot(cumulative_variance)
        # Calculate the index where cumulative variance exceeds or meets 95%
        threshold = self.pc_percentage
        threshold_index = next(
            x[0] for x in enumerate(cumulative_variance) if x[1] >= 1 - threshold
        )

        # Add a horizontal dashed line at 95% threshold
        plt.axvline(
            x=threshold + 1,
            color="r",
            linestyle="--",
            label=f"{self.pc_percentage*100}% Contribution at PC {threshold_index+1}",
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
        return f"Scree Plot saved as {plot_id} ID\n" + extra_mess

    def make_pc_plots(self):
        self._make_df()
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
        return f"PCA plots saved as {plot_id} "

    def measure_cosine_convergence(self):
        if not self.pc:
            pc_mess = self.get_pc()
        else:
            pc_mess = ""
        if not self.n_pcs:
            self._get_number_pcs()
            self._make_transformation()

        pc_messages = []
        for i in range(self.n_pcs):
            cc = pca.cosine_content(self.transformed, i)
            pc_messages.append(f"Cosine Content for PC {i+1}-{cc:.3f}\n")
        cc_message = f"Cosine Content of each PC: {','.join(pc_messages)}"
        return pc_mess + cc_message

    def run_all(self):
        self.get_pc()
        try:
            scree_plot = self.make_scree_plot()
        except Exception as e:
            raise Exception(f"Error during Scree Plot: str({e})")
        try:
            pc_plots = self.make_pc_plots()
        except Exception as e:
            raise Exception(f"Error during pc plots: str({e})")
        try:
            cos_conv = self.measure_cosine_convergence()
        except Exception as e:
            raise Exception(f"Error during cosine convergence: str({e})")
        return f"Analyses done: {scree_plot}, {pc_plots}, {cos_conv}"


class PCASchema(BaseModel):
    trajectory_fileid: str = Field(
        description="Trajectory File ID of the simulation to be analyzed"
    )
    topology_fileid: str = Field(
        description=("Topology File ID of " "the simulation to be analyzed")
    )
    pc_percentage: Optional[float] = Field(
        95.0, description="Max cumulative percentage of components for analysis"
    )
    analysis: str = Field(
        "all",
        description=(
            "Type of analysis to be done. Availables are: "
            "scree_plot (Saves a scree plot of the eigenvalues), "
            "pca_analysis (gets principal components and saves"
            " a grid plot for visualization) "
            "Cosine (measures the cosine convergence of the top 3 PCs)"
            "all (makes all of the previous analysis)"
        ),
    )
    selection: Optional[str] = Field(
        "backbone and name CA",
        description=(
            "Which selection of atoms from the simulation "
            "to use for the pca analysis"
        ),
    )
    remove_terminals: Optional[str] = Field(
        False, description="To remove or not the terminal residues of each chain."
    )


class PCATool(BaseTool):
    name = "PCATool"
    description = (
        "Calculate the Principal Analysis Components of a MD trajectory and "
        "performs analysis with them"
    )
    args_schema = PCASchema
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry] = None):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, input):
        try:
            input = self.validate_input(**input)
        except ValueError as e:
            return f"Error using the PCA Tool: {str(e)}"

        error = input.get("error", None)
        if error:
            return f"Error with the tool inputs: {error} "
        system_input_message = input.get("system_message")
        if system_input_message == "Tool Messages:":
            system_input_message = ""
        traj_id = input.get("trajectory_fileid")
        top_id = input.get("topology_fileid")
        traj_path = self.path_registry.get_mapped_path(traj_id)
        top_path = self.path_registry.get_mapped_path(top_id)
        pc_percentage = input.get("pc_percentage")
        analysis = input.get("analysis")
        selection = input.get("selection", "backbone and name CA")

        PCA_container = PCA_analysis(
            self.path_registry,
            pc_percentage=pc_percentage,
            top_path=top_path,
            traj_path=traj_path,
            sim_id=traj_id,
            selection=selection,
        )
        try:
            if analysis == "all":
                result = PCA_container.run_all()
                return f"{result}. \n\n {system_input_message}"
            if analysis.lower() == "cosine_convergence":
                result = PCA_container.measure_cosine_convergence()
                return f"Analyses done: {result}. \n\n {system_input_message}"
            if analysis.lower() == "scree_plot":
                result = PCA_container.make_scree_plot()
                return f"Analyses done: {result}. \n\n {system_input_message}"
        except Exception as e:
            raise (Exception(f"Error during PCA Tool usage: {str(e)}"))

    def validate_input(self, **input):
        input = input.get("action_input", input)
        input = input.get("input", input)
        trajectory_id = input.get("trajectory_fileid", None)
        topology_id = input.get("topology_fileid", None)
        pc_percentage = input.get("pc_percentage", 95.0)
        analysis = input.get("analysis", "all")
        selection = input.get("selection", "name CA")
        remove_terminals = input.get("remove_terminals", False)
        if not trajectory_id:
            raise ValueError("Incorrect Inputs: trajectory_fileid is required")
        if not topology_id:
            raise ValueError("Incorrect Inputs: topology_fileid is required")
        # check if trajectory id is valid
        fileids = self.path_registry.list_path_names()
        error = ""
        system_message = "Tool Messages:"
        print("Files Ids:", fileids)
        if trajectory_id not in fileids:
            error += " Trajectory File ID not in path registry"
        if topology_id not in fileids:
            error += " Topology File ID not in path registry"
        try:
            pc_percentage = float(pc_percentage)
        except ValueError as e:
            if "%" in str(e):
                pc_percentage.replace("%", "")
                try:
                    pc_percentage = float(pc_percentage)
                except Exception as e:
                    error += " pc_percentage value must be a float"
            else:
                error += " pc_percentage value must be a float"
        except Exception:
            error += " pc_percentage value must be a float"

        if analysis.lower() not in [
            "all",
            "pca_analysis",
            "scree_plot",
            "cosine_convergence",
        ]:
            analysis = "all"
            system_message += (
                " analysis arg not recognized, using analysis = 'all' as default"
            )

        if selection not in [
            "backbone",
            "name CA",
            "backbone and name CA",
            "protein",
            "all",
        ]:
            selection = "name CA"  # just alpha carbons
        # get all the kwargs:
        keys = input.keys()
        for key in keys:
            if key not in [
                "trajectory_fileid",
                "topology_fileid",
                "pc_percentage",
                "analysis",
                "selection",
                "remove_terminals",
            ]:
                system_message += f"{key} is not part of admitted tool inputs"
        if error == "":
            error = None
        return {
            "trajectory_fileid": trajectory_id,
            "topology_fileid": topology_id,
            "pc_percentage": pc_percentage,
            "analysis": analysis,
            "remove_terminals": remove_terminals,
            "selection": selection,
            "error": error,
            "system_message": system_message,
        }
