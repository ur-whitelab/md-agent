from typing import Optional

import mdtraj as md
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from mdagent.utils import PathRegistry


class SaltBridgeFunction:  # this class defines a method called find_salt_bridge
    # using MD traj and top files and threshold distance default, residue pair list
    path_registry:PathRegistry
    def __init__(self, path_registry):
        self.path_registry = path_registry
        self.salt_bridges = []  # stores paired salt bridges
        self.traj = None

    def find_salt_bridges(
        self, traj_file, top_file=None, threshold_distance=0.4, residue_pairs=None
    ):
        # add two files here in similar format as line 14 above
        traj_file_path = self.path_registry.get_mapped_path(traj_file)
        ending = traj_file_path.split(".")[-1]
        if ending in ["dcd", "xtc", "xyz"] and top_file is not None:
            top_file_path = self.path_registry.get_mapped_path(top_file)
            self.traj = md.load(traj_file_path, top=top_file_path)
        else:
            self.traj = md.load(traj_file_path)

        if residue_pairs is None:
            residue_pairs = [
                ("ARG", "ASP"),
                ("ARG", "GLU"),
                ("LYS", "ASP"),
                ("LYS", "GLU"),
            ]

        for pair in residue_pairs:
            donor_residues = self.traj.topology.select(f'residue_name == "{pair[0]}"')
            acceptor_residues = self.traj.topology.select(
                f'residue_name == "{pair[1]}"'
            )

            for donor_idx in donor_residues:
                for acceptor_idx in acceptor_residues:
                    distances = md.compute_distances(
                        self.traj, [[donor_idx, acceptor_idx]]
                    )
                    if any(d <= threshold_distance for d in distances):
                        self.salt_bridges.append((donor_idx, acceptor_idx))
        return self.salt_bridges

    def get_results_string(self):
        msg = "Salt bridges found: "
        for bridge in self.salt_bridges:
            msg += (
                f"Residue {self.traj.topology.atom(bridge[0]).residue.index + 1} "
                f"({self.traj.topology.atom(bridge[0]).residue.name}) - "
                f"Residue {self.traj.topology.atom(bridge[1]).residue.index + 1} "
                f"({self.traj.topology.atom(bridge[1]).residue.name})"
            )
        return msg


class SaltBridgeToolInput(BaseModel):
    trajectory_fileid: str = Field(
        None, description="Trajectory file ID. Either dcd, hdf5, xtc, or xyz"
    )

    topology_fileid: Optional[str] = Field(None, description="Topology file ID")

    threshold_distance: Optional[float] = Field(
        0.4,
        description=(
            "maximum distance between residues for salt bridge formation in angstrom"
        ),
    )

    residue_pairs: Optional[dict] = Field(
        None, description=("Identifies the amino acid residues for salt bridge")
    )


class SaltBridgeTool(BaseTool):
    name = "SaltBridgeTool"
    description = "A tool to find salt bridge in a protein trajectory"
    args_schema = SaltBridgeToolInput
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry):
        super().__init__()
        self.path_registry = path_registry

    def _run(
        self, traj_file, top_file=None, threshold_distance=0.4, residue_pairs=None
    ):
        try:
            # calls the salt bridge function
            salt_bridge_function = SaltBridgeFunction(self.path_registry)
            salt_bridge_function.find_salt_bridges(
                traj_file, top_file, threshold_distance, residue_pairs
            )
            message = salt_bridge_function.get_results_string()
        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"
        return "Succeeded. " + message
