import itertools
from typing import Optional

import mdtraj as md
import numpy as np
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from mdagent.utils import PathRegistry

from .descriptions import (
    CONTACT_SELECTION_DESC,
    CONTACTS_TOOL_DESC,
    CUTOFF_DESC,
    DISPLACEMENT_TOOL_DESC,
    DISTANCE_TOOL_DESC,
    NEIGHBORS_TOOL_DESC,
    SELECTION_DESC,
    TOPOLOGY_FILEID_DESC,
    TRAJECTORY_FILEID_DESC,
)


class distanceUtils:
    def all_possible_pairs(self, array1, array2):
        return list(itertools.product(array1, array2))

    def calc_residue_dist(self, residues=(0, 0)):
        """
        Return the C-alpha distance between two residues.
        """
        diff_vector = residues[0] - residues[1]
        return np.sqrt(np.vdot(diff_vector, diff_vector))

    def calc_side_center_mass(self, traj, topology, frame):
        """
        Return approximate center of mass of each side chain.
        COM od Glycine is approximated by the coordinate of its CA atom.
        """
        scmass = []

        for i in range(0, topology.n_residues):
            selection = topology.select("(resid %d) and sidechain" % i)
            if len(selection) < 1:
                selection = topology.select("(resid %d) and (name CA)" % i)
            mass = np.array([0.0, 0.0, 0.0])
            for atom in selection:
                mass += traj.xyz[frame, atom, :]
            scmass.append(mass / len(selection))
        return scmass

    def calc_residue_side_dist(self, traj, frame, residue_one, residue_two):
        """
        Return the C-alpha distance between two residues.
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

    def calc_matrix_cm(self, traj, frame, threshold=0.8, distance=1.2):
        """
        Used internally to compute the matrix data when the object is
        initialized.
        """
        selection = traj.topology.select("name CA")
        scmass = self.calc_side_center_mass(traj.topology, frame)
        dim = len(selection)

        matrix = np.zeros(dim, dim)
        for row, atom1 in enumerate(selection):
            for col, atom2 in enumerate(selection):
                # Only calculate once
                if col > row - 1:
                    continue

                # the atom pointprint(traj.xyz[0, atom, :])
                val = self.calc_residue_dist(
                    residues=(traj.xyz[frame, atom1, :], traj.xyz[frame, atom2, :])
                )
                # Center of mass

                dis = self.calc_residue_dist((scmass[col], scmass[row]))

                matrix[row, col] = val < threshold

                if dis < distance:
                    matrix[col, row] = 1 - dis / distance
                else:
                    matrix[col, row] = 0

        return matrix

    def calc_matrix_dis(self, traj, frame, threshold=0.8, distance=(0.5, 1.2)):
        """
        Used internally to compute the matrix data when the object is
        initialized.
        """
        selection = traj.topology.select("name CA")
        self.calc_side_center_mass(traj.topology, frame)
        len(selection)

        matrix = np.zeros((len(selection), len(selection)))  # , np.bool)
        for row, atom1 in enumerate(selection):
            for col, atom2 in enumerate(selection):
                # Only calculate once
                if col > row - 1:
                    continue

                # the atom pointprint(traj.xyz[0, atom, :])
                val = self.calc_residue_dist(
                    residues=(traj.xyz[frame, atom1, :], traj.xyz[frame, atom2, :])
                )
                # closest distance of sidechains
                dis = self.calc_residue_side_dist(frame, row, col)

                matrix[row, col] = val < threshold

                if dis < distance[0]:
                    matrix[col, row] = 1
                elif distance[0] < dis < distance[1]:
                    matrix[col, row] = (distance[1] - dis) / (distance[1] - distance[0])
        return matrix


class distanceSchema(BaseModel):
    trajectory_fileid: str = Field(description=TRAJECTORY_FILEID_DESC)
    topology_fileid: str = Field(description=TOPOLOGY_FILEID_DESC)
    analysis: str = Field(
        "all", description="Which residues to calculate distance from"
    )
    mode: str = Field(
        "CA",
        description=(
            "What to use for distance calculation, either "
            "alpha carbons (CA) or center of mass (COM)"
        ),
    )
    selection1: str = Field(description="First" + SELECTION_DESC)
    selection2: str = Field(description="Second" + SELECTION_DESC)


class displacementSchema(BaseModel):
    trajectory_fileid: str = Field(description=TRAJECTORY_FILEID_DESC)
    topology_fileid: str = Field(description=TOPOLOGY_FILEID_DESC)
    selection1: str = Field(description="First" + SELECTION_DESC)
    selection2: str = Field(description="Second" + SELECTION_DESC)


class neighborsSchema(BaseModel):
    trajectory_fileid: str = Field(description=TRAJECTORY_FILEID_DESC)
    topology_fileid: str = Field(description=TOPOLOGY_FILEID_DESC)
    selection: str = Field(description=SELECTION_DESC)
    cutoff: float = Field(10.0, description=CUTOFF_DESC)


class contactSchema(BaseModel):
    trajectory_fileid: str = Field(description=TRAJECTORY_FILEID_DESC)
    topology_fileid: str = Field(description=TOPOLOGY_FILEID_DESC)
    selection: str = Field(description=CONTACT_SELECTION_DESC)
    cutoff: float = Field(6.0, description=CUTOFF_DESC)


class distanceTool(BaseTool):
    name = "distanceTool"
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

        path_to_traj = self.path_registry.get_mapped_path(trajectory_id)
        path_to_top = self.path_registry.get_mapped_path(topology_id)
        traj = md.load(path_to_traj, top=path_to_top)
        atom_indices1 = traj.top.select(selection1)
        atom_indices2 = traj.top.select(selection2)
        utils = distanceUtils()
        pairs = utils.all_possible_pairs(atom_indices1, atom_indices2)
        md.compute_distances(traj, pairs)

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
                "analysis",
                "mode",
                "selection1",
                "selection2",
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
        cutoff = input.get("cutoff", 10.0)
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

    def _run(self):
        pass

    def _arun(self):
        pass

    def validate_inputs(self, input):
        input = input.get("action_input", input)
        input = input.get("input", input)
        trajectory_id = input.get("trajectory_fileid", None)
        topology_id = input.get("topology_fileid", None)
        selection = input.get("selection", "all")
        cutoff = input.get("cutoff", 6.0)
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
