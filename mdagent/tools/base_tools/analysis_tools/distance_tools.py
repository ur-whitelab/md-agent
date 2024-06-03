import itertools
import os
from typing import Literal, Optional

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import pandas as pd
from langchain.tools import BaseTool
from matplotlib.animation import FuncAnimation
from pydantic import BaseModel, Field

from mdagent.utils import FileType, PathRegistry

from .descriptions import (
    CONTACT_SELECTION_DESC,
    CONTACTS_TOOL_DESC,
    CUTOFF_DESC,
    DISPLACEMENT_TOOL_DESC,
    DISTANCE_TOOL_DESC,
    NEIGHBORS_TOOL_DESC,
    RES_SELECTION_DESC,
    SELECTION_DESC,
    TOPOLOGY_FILEID_DESC,
    TRAJECTORY_FILEID_DESC,
)


class distanceToolsUtils:
    def __init__(self, path_registry: Optional[PathRegistry] = None):
        self.path_registry = path_registry

    def all_possible_pairs(self, array1, array2):
        return list(itertools.product(array1, array2))

    def calc_residue_dist(self, residues=(0, 0)):
        """
        Return the C-alpha distance between two residues.
        returns: float  distance
        """
        diff_vector = residues[0] - residues[1]
        return np.sqrt(np.vdot(diff_vector, diff_vector))

    def calc_side_center_mass(self, traj):
        """
        Return approximate center of mass of each side chain per frame.
        COM od Glycine is approximated by the coordinate of its CA atom.

        Note: Because compute_center_of_mass gets the center of mass of only one
        selection at a time, we need to loop over all residues to get the each COM.

        returns: com_matrix: np.array, shape=(n_frames, n_residues, 3)
        """
        traj = traj.atom_slice(traj.top.select("protein"), inplace=False)
        com_matrix = np.zeros((traj.n_frames, traj.n_residues, 3))
        for i in range(0, traj.topology.n_residues):
            selection = f"(resid {i}) and sidechain"
            if len(selection) < 1:
                selection = f"(resid {i}) and (name CA)"
            try:
                com_matrix[:, i, :] = md.compute_center_of_mass(traj, select=selection)
            except Exception as e:
                raise (f"Error calculating center of mass for residue, {i}: {str(e)}")
        return com_matrix

    def calc_dis_matrix_from_com_all_resids(self, traj, com_matrix):
        """
        Return the distance matrix between the center of mass of residues

        returns: dis_matrix: np.array, shape=(n_frames, n_residues, n_residues)
        """
        new_xyz = com_matrix
        new_topology = pd.DataFrame(
            {
                "serial": range(traj.n_residues),
                "name": ["COM" for _ in traj.topology.residues],
                "resSeq": range(traj.n_residues),
                "resName": [i.name for i in traj.topology.residues],
                "element": ["VS" for _ in traj.topology.residues],
                "chainID": [
                    i.chain.chain_id if i.chain.chain_id else 0
                    for i in traj.topology.residues
                ],
                "segmentID": [i.segment_id for i in traj.topology.residues],
            }
        )
        print(new_topology.head())
        top = md.Topology.from_dataframe(new_topology)
        new_traj = md.Trajectory(new_xyz, top)
        pairs = self.all_possible_pairs(range(traj.n_residues), range(traj.n_residues))
        _dis_matrix = md.compute_distances(new_traj, pairs)
        dis_matrix = md.geometry.squareform(_dis_matrix, residue_pairs=pairs)
        return dis_matrix

    def calc_residue_side_dist(self, traj, frame, residue_one, residue_two):
        """
        Return minimum distance between two residues side-chains.
        returns: float  distance
        """
        # Select first residue
        selection1 = traj.topology.select("(resid %d) and sidechain" % residue_one)
        if len(selection1) < 1:
            selection1 = traj.topology.select("(resid %d) and (name CA)" % residue_one)
        # Select second residue
        selection2 = traj.topology.select("(resid %d) and sidechain" % residue_two)
        if len(selection2) < 1:
            selection2 = traj.topology.select("(resid %d) and (name CA)" % residue_two)
        atom_pairs = self.all_possible_pairs(
            traj.xyz[frame, selection1, :], traj.xyz[frame, selection2, :]
        )
        return min(map(self.calc_residue_dist, atom_pairs))

    def calc_matrix_cm_all_resids(traj, threshold=0.8, distance=1.2):
        """
        Used internally to compute the matrix of contacts between residues

        returns: matrix: np.array, shape=(n_frames, n_residues, n_residues)
        """

        # By default, this ↓ ignores any non-protein atoms
        distances, residue_pairs = md.compute_contacts(traj, scheme="closest-heavy")
        matrix = md.geometry.squareform(distances, residue_pairs=residue_pairs)
        for frame in range(matrix.shape[0]):
            for i in range(matrix.shape[1]):
                for j in range(matrix.shape[2]):
                    if matrix.shape[0] == matrix.shape[1]:
                        if i > j:
                            if matrix[frame, i, j] < threshold:
                                matrix[frame, i, j] = 1
                            else:
                                matrix[frame, i, j] = 0
                        else:
                            if matrix[frame, i, j] < threshold:
                                matrix[frame, i, j] = 1
                            elif matrix[frame, i, j] > distance:
                                matrix[frame, i, j] = 0
                            else:
                                matrix[frame, i, j] = 1 - (
                                    matrix[frame, i, j] - threshold
                                ) / (distance - threshold)
                    else:
                        return matrix < threshold
        return matrix

    def calc_matrix_dis_ca_all_resids(self, traj):
        """
        Gets the Matrix distance between all alpha carbons of residues
        returns: dis_matrix: np.array, shape=(n_frames, n_residues, n_residues)
        """
        self.calc_side_center_mass(traj)
        traj = traj.atom_slice(traj.top.select("protein"))
        residues = traj.topology.select("name CA")
        residues_ids = np.arange(0, traj.topology.n_residues)
        residue_pairs = self.all_possible_pairs(residues, residues)
        # needed to use squareform!
        residue_id_pairs = self.all_possible_pairs(residues_ids, residues_ids)
        distances = md.compute_distances(traj, residue_pairs)
        dis_matrix = md.geometry.squareform(distances, residue_pairs=residue_id_pairs)

        return dis_matrix

    def save_matrix_frame(self, matrix, path):
        """
        Saves the distance matrix as images in the path specified. It samples 10 frames
        from the matrix and saves them as images.
        """
        if not os.path.exists(path):
            os.makedirs(path)

        for i in range(0, matrix.shape[0], matrix.shape[0] // 10):
            fig, ax = plt.subplots()
            ax.imshow(matrix[i], origin="lower")
            ax.set_title(f"Distance Matrix frame {i}")

            plt.savefig(f"{path}/dist_matrix_frame_{i}.png")

    def make_movie(
        self, matrix, option="distance", source_id="unkown", path="distance.gif"
    ):
        """
        Create an animation of the distance matrix over time
        saves: a gif of the animation
        returns: None
        """
        if option == "distance":
            Title = "Distance Matrix"
        elif option == "contact":
            Title = "Contact Matrix"
        fig, ax = plt.subplots()
        im = ax.imshow(matrix[0], origin="lower")

        # Create a title text object
        title_text = ax.set_title(f"{Title} frame {0}")

        def update(frame):
            im.set_array(matrix[frame])
            # Update the title
            title_text.set_text(f"{Title} frame {frame}")
            return im, title_text  # Return both the image and the title text object

        ani = FuncAnimation(fig, update, frames=matrix.shape[0], interval=200)
        ani.save(path)
        description = f"{option} matrix over time with {matrix.shape[0]} \
                frames, and {matrix.shape[1]} residues. \
                trajectory file: {source_id}"
        return description


class distanceSchema(BaseModel):
    trajectory_fileid: str = Field(description=TRAJECTORY_FILEID_DESC)
    topology_fileid: str = Field(description=TOPOLOGY_FILEID_DESC)
    analysis: Literal["all", "not all"] = Field(
        "all",
        description=(
            "Which residues ids to calculate distance from, if all, "
            "all residues will be used if not all, only the selected"
            " residues (selection1 and selection2) will be used"
        ),
    )
    mode: Literal["CA", "COM"] = Field(
        "CA",
        description=(
            "What to use for distance calculation, either "
            "alpha carbons (CA) or center of mass (COM)"
        ),
    )
    selection1: Optional[str] = Field(description="First" + RES_SELECTION_DESC)
    selection2: Optional[str] = Field(description="Second" + RES_SELECTION_DESC)


class displacementSchema(BaseModel):
    trajectory_fileid: str = Field(description=TRAJECTORY_FILEID_DESC)
    topology_fileid: str = Field(description=TOPOLOGY_FILEID_DESC)
    selection1: str = Field(description="First" + SELECTION_DESC)
    selection2: str = Field(description="Second" + SELECTION_DESC)


class neighborsSchema(BaseModel):
    trajectory_fileid: str = Field(description=TRAJECTORY_FILEID_DESC)
    topology_fileid: str = Field(description=TOPOLOGY_FILEID_DESC)
    selection: str = Field(description=SELECTION_DESC)
    cutoff: float = Field(1.0, description=CUTOFF_DESC)


class contactSchema(BaseModel):
    trajectory_fileid: str = Field(description=TRAJECTORY_FILEID_DESC)
    topology_fileid: str = Field(description=TOPOLOGY_FILEID_DESC)
    selection: str = Field(description=CONTACT_SELECTION_DESC)
    cutoff: float = Field(0.8, description=CUTOFF_DESC)


class distanceMatrixTool(BaseTool):
    name = "distanceMatrixTool"
    description = DISTANCE_TOOL_DESC
    input_schema = distanceSchema
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry] = None):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, **input):
        try:
            input = self.validate_input(**input)
        except ValueError as e:
            return f"Error using the PCA Tool: {str(e)}"

        error = input.get("error", None)
        if error:
            return f"Error with the tool inputs: {error} "
        input.get("system_message")

        trajectory_id = input["trajectory_fileid"]
        topology_id = input["topology_fileid"]
        selection1 = input["selection1"]
        selection2 = input["selection2"]
        mode = input["mode"]
        analysis = input["analysis"]
        path_to_traj = self.path_registry.get_mapped_path(trajectory_id)
        path_to_top = self.path_registry.get_mapped_path(topology_id)
        traj = md.load(path_to_traj, top=path_to_top)
        if analysis != "all":
            # slice the trajectory to include only the selections
            atom_indices1 = traj.top.select(selection1)
            atom_indices2 = traj.top.select(selection2)
            traj = traj.atom_slice(atom_indices1 + atom_indices2, inplace=False)
        utils = distanceToolsUtils(path_registry=self.path_registry)
        if mode == "CA":
            dist_matrix = utils.calc_matrix_dis_ca_all_resids(traj)
        elif mode == "COM":
            # calculates distances matrix using center of mass of side chains
            com_matrix = utils.calc_side_center_mass(traj)
            dist_matrix = utils.calc_dis_matrix_from_com_all_resids(traj, com_matrix)

        # plotting the distance matrix
        path = f"{self.path_registry.ckpt_figures}/dist_{trajectory_id}/\
            dist_matrix_{trajectory_id}.gif"
        if not os.path.exists(
            f"{self.path_registry.ckpt_figures}/dist_{trajectory_id}"
        ):
            os.makedirs(f"{self.path_registry.ckpt_figures}/dist_{trajectory_id}")
        fig_id = self.path_registry.get_fileid(file_name=path, type=FileType.FIGURE)
        movie_desc = utils.make_movie(dist_matrix, path=path, source_id=trajectory_id)
        self.path_registry.map_path(fig_id, path, movie_desc)
        # save some of the frames as images
        utils.save_matrix_frame(
            dist_matrix, f"{self.path_registry.ckpt_figures}/dist_{trajectory_id}"
        )

        return "Distance Matrix created with ID: " + fig_id

    def _arun(self):
        pass

    def validate_inputs(self, input):
        input = input.get("action_input", input)
        input = input.get("input", input)
        trajectory_id = input.get("trajectory_fileid", None)
        topology_id = input.get("topology_fileid", None)
        selection1 = input.get("selection1", "name CA")
        selection2 = input.get("selection2", "name CA")
        mode = input.get("mode", "CA")
        if not trajectory_id:
            raise ValueError("Incorrect Inputs: trajectory_fileid is required")
        if not topology_id:
            raise ValueError("Incorrect Inputs: topology_fileid is required")
        fileids = self.path_registry.list_path_names()
        error = ""
        system_message = ""
        if trajectory_id not in fileids:
            error += " Trajectory File ID not in path registry"
        if topology_id not in fileids:
            error += " Topology File ID not in path registry"
        if mode not in ["CA", "COM"]:
            system_message += " Incorrect mode, must be either CA or COM.\
                  Defaulting to CA \n"
            mode = "CA"
        keys = input.keys()
        for key in keys:
            if key not in [
                "trajectory_fileid",
                "topology_fileid",
                "analysis",
                "mode",
                "selection1",
                "selection2",
            ]:
                system_message += f"{key} is not part of admitted tool inputs\n"

        if error == "":
            error = None
        return {
            "trajectory_fileid": trajectory_id,
            "topology_fileid": topology_id,
            "selection1": selection1,
            "selection2": selection2,
            "error": error,
            "system_message": system_message,
        }


class displacementTool(BaseTool):
    name = "displacementTool"
    description = DISPLACEMENT_TOOL_DESC
    input_schema = displacementSchema
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry] = None):
        super().__init__()
        self.path_registry = path_registry

    def _run(self):
        pass

    def _arun(self):
        pass

    def validate_inputs(self, input):
        input = input.get("action_input", input)
        input = input.get("input", input)
        trajectory_id = input.get("trajectory_fileid", None)
        topology_id = input.get("topology_fileid", None)
        selection1 = input.get("selection1", "name CA")
        selection2 = input.get("selection2", "name CA")
        if not trajectory_id:
            raise ValueError("Incorrect Inputs: trajectory_fileid is required")
        if not topology_id:
            raise ValueError("Incorrect Inputs: topology_fileid is required")
        fileids = self.path_registry.list_path_names()
        error = ""
        system_message = ""
        if trajectory_id not in fileids:
            error += " Trajectory File ID not in path registry"
        if topology_id not in fileids:
            error += " Topology File ID not in path registry"
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
            "selection1": selection1,
            "selection2": selection2,
            "error": error,
            "system_message": system_message,
        }


class neighborsTool(BaseTool):
    name = "neighborsTool"
    description = NEIGHBORS_TOOL_DESC
    input_schema = neighborsSchema
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry] = None):
        super().__init__()
        self.path_registry = path_registry

    def _run(self):
        pass

    def _arun(self):
        pass

    def validate_inputs(self, input):
        input = input.get("action_input", input)
        input = input.get("input", input)
        trajectory_id = input.get("trajectory_fileid", None)
        topology_id = input.get("topology_fileid", None)
        selection = input.get("selection", "name CA")
        cutoff = input.get("cutoff", 1.0)
        if not trajectory_id:
            raise ValueError("Incorrect Inputs: trajectory_fileid is required")
        if not topology_id:
            raise ValueError("Incorrect Inputs: topology_fileid is required")
        fileids = self.path_registry.list_path_names()
        error = ""
        system_message = ""
        if trajectory_id not in fileids:
            error += " Trajectory File ID not in path registry"
        if topology_id not in fileids:
            error += " Topology File ID not in path registry"
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
            "selection": selection,
            "cutoff": cutoff,
            "error": error,
            "system_message": system_message,
        }


class contactsTool(BaseTool):
    name = "NeighborsTool"
    description = CONTACTS_TOOL_DESC
    input_schema = neighborsSchema
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry] = None):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, **input):
        input = self.validate_input(**input)
        error = input.get("error", None)
        if error:
            return f"Error with the tool inputs: {error} "
        input.get("system_message")
        trajectory_id = input["trajectory_fileid"]
        topology_id = input["topology_fileid"]
        selection = input["selection"]
        cutoff = input["cutoff"]
        system_message = input["system_message"]
        path_to_traj = self.path_registry.get_mapped_path(trajectory_id)
        path_to_top = self.path_registry.get_mapped_path(topology_id)
        traj = md.load(path_to_traj, top=path_to_top)
        if selection != "all":
            try:
                atom_indices = traj.top.select(selection)
                traj = traj.atom_slice(atom_indices, inplace=False)
            except Exception as e:
                system_message += f"Error with the selection: {str(e)}.\
                      Defaulting to 'all'"

        utils = distanceToolsUtils(path_registry=self.path_registry)
        matrix = utils.calc_matrix_cm_all_resids(traj, threshold=cutoff)

        # save the matrix as a gif and some of the frames as images
        path = f"{self.path_registry.ckpt_figures}/contact_{trajectory_id}/\
            contact_matrix_{trajectory_id}.gif"
        if not os.path.exists(
            f"{self.path_registry.ckpt_figures}/contact_{trajectory_id}"
        ):
            os.makedirs(f"{self.path_registry.ckpt_figures}/contact_{trajectory_id}")

        fig_id = self.path_registry.get_fileid(file_name=path, type=FileType.FIGURE)
        movie_desc = utils.make_movie(matrix, path=path, source_id=trajectory_id)
        self.path_registry.map_path(fig_id, path, movie_desc)
        utils.save_matrix_frame(
            matrix, f"{self.path_registry.ckpt_figures}/contact_{trajectory_id}"
        )
        return "Contact Matrix Figure created with ID: " + fig_id

    def _arun(self):
        pass

    def validate_inputs(self, input):
        input = input.get("action_input", input)
        input = input.get("input", input)
        trajectory_id = input.get("trajectory_fileid", None)
        topology_id = input.get("topology_fileid", None)
        selection = input.get("selection", "all")
        cutoff = input.get("cutoff", 0.8)
        if not trajectory_id:
            raise ValueError("Incorrect Inputs: trajectory_fileid is required")
        if not topology_id:
            raise ValueError("Incorrect Inputs: topology_fileid is required")
        fileids = self.path_registry.list_path_names()
        error = ""
        system_message = ""
        if trajectory_id not in fileids:
            error += " Trajectory File ID not in path registry"
        if topology_id not in fileids:
            error += " Topology File ID not in path registry"
        keys = input.keys()
        for key in keys:
            if key not in [
                "trajectory_fileid",
                "topology_fileid",
                "cutoff",
                "selection",
            ]:
                system_message += f"{key} is not part of admitted tool inputs"
        if error == "":
            error = None
        return {
            "trajectory_fileid": trajectory_id,
            "topology_fileid": topology_id,
            "selection": selection,
            "cutoff": cutoff,
            "error": error,
            "system_message": system_message,
        }
