from typing import Optional

import mdtraj as md
import numpy as np
import pandas as pd
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from mdagent.utils import FileType, PathRegistry, load_single_traj


class SaltBridgeFunction:
    def __init__(self, path_registry):
        self.path_registry = path_registry
        self.salt_bridges = []  # stores paired salt bridges
        self.traj = None

    def find_salt_bridges(
        self,
        traj,
        threshold_distance: float = 0.4,
        residue_pairs=None,
    ):
        """Find Salt Bridge in molecular dynamics simulation trajectory.

        Description:
        traj: MDtraj trajectory
        thresold_distance: maximum distance between residues for salt bridge formation
        residue_pairs: list of pairs of residues
        """
        if traj is None:
            raise Exception("Trajectory is None")

        self.traj = traj
        if residue_pairs is None:
            residue_pairs = [
                ("ARG", "ASP"),
                ("ARG", "GLU"),
                ("LYS", "ASP"),
                ("LYS", "GLU"),
            ]
        donor_acceptor_pairs = []

        for pair in residue_pairs:
            print(f"Looking for salt bridges between {pair[0]} and {pair[1]} pairs...")

            donor_residues = self.traj.topology.select(f'residue_name == "{pair[0]}"')
            acceptor_residues = self.traj.topology.select(
                f'residue_name == "{pair[1]}"'
            )
            # generate all possible donor-acceptor pairs
            pairs = np.array(np.meshgrid(donor_residues, acceptor_residues)).T.reshape(
                -1, 2
            )
            donor_acceptor_pairs.append(pairs)

            # Combines all rsidue pairs
            donor_acceptor_pairs = np.vstack(donor_acceptor_pairs)
            # filter by threshold distance

            all_distance = md.compute_distances(self.traj, donor_acceptor_pairs)

            mini_distances = np.min(all_distance, axis=0)
            within_threshold = mini_distances <= threshold_distance
            filtered_pairs = donor_acceptor_pairs[within_threshold]

            self.salt_bridges = [tuple(pair) for pair in filtered_pairs]
            file_id = self.save_results_to_file()

            return file_id

    def save_results_to_file(self):
        if self.traj is None:
            raise Exception("Trajectory is None")

        salt_bridge_data = []

        for bridge in self.salt_bridges:
            donor_atom = self.traj.topology.atom(bridge[0])
            acceptor_atom = self.traj.topology.atom(bridge[1])
            salt_bridge_data.append(
                {
                    "Donor Atom Index": bridge[0],
                    "Donor Residue": (
                        f"{donor_atom.residue.index +1} "
                        f" ({donor_atom.residue.name})"
                    ),
                    "Acceptor Atom Index": bridge[1],
                    "Acceptor Atom Residue": (
                        f"{acceptor_atom.residue.index + 1} "
                        f"({acceptor_atom.residue.name})"
                    ),
                }
            )

        df = pd.DataFrame(salt_bridge_data)

        file_name = self.path_registry.write_file_name(
            FileType.RECORD,
            record_type="salt_bridges",
            file_format="csv",
        )
        file_id = self.path_registry.get_fileid(file_name, FileType.RECORD)
        file_path = f"{self.path_registry.ckpt_records}/{file_name}"
        df.to_csv(file_path, index=False)
        self.path_registry.map_path(
            file_id, file_path, description="salt bridge analysis"
        )
        return file_id

    def get_results_string(self):
        msg = "Salt bridges found: "
        for bridge in self.salt_bridges:
            msg += (
                f"Residue {self.traj.topology.atom(bridge[0]).residue.index + 1} "
                f"({self.traj.topology.atom(bridge[0]).residue.name}) - "
                f"Residue {self.traj.topology.atom(bridge[1]).residue.index + 1} "
                f"({self.traj.topology.atom(bridge[1]).residue.name})\n"
            )
        return msg


class SaltBridgeToolInput(BaseModel):
    traj_id: str = Field(
        None,
        description="Trajectory file ID. Either dcd, hdf5, xtc, or xyz",
    )

    top_id: Optional[str] = Field(None, description="Topology file ID")

    threshold_distance: Optional[float] = Field(
        0.4,
        description=(
            "maximum distance between residues for salt bridge formation in angstrom"
        ),
    )

    residue_pairs: Optional[str] = Field(
        None,
        description=("Identifies the amino acid residues for salt bridge"),
    )


class SaltBridgeTool(BaseTool):
    name = "SaltBridgeTool"
    description = "A tool to find salt bridge in a protein trajectory"
    args_schema = SaltBridgeToolInput
    path_registry: PathRegistry | None = None

    def __init__(self, path_registry):
        super().__init__()
        self.path_registry = path_registry

    def _run(
        self,
        traj_file: str,
        top_file: str | None = None,
        threshold_distance=0.4,
        residue_pairs=None,
    ):
        try:
            if self.path_registry is None:
                return "Path registry is not set"
            traj = load_single_traj(self.path_registry, traj_file, top_file)
            if not traj:
                return "Trajectory Failed to load ."

            # Load trajectory using MDTraj

            traj = load_single_traj(traj_file, top_file)
            # calls the salt bridge function
            salt_bridge_function = SaltBridgeFunction(self.path_registry)
            results_file_id = salt_bridge_function.find_salt_bridges(
                traj, threshold_distance, residue_pairs
            )
            message = salt_bridge_function.get_results_string()
            message += f"Saved to results file with fle id: {results_file_id}"
            return "Succeeded. " + message
        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"
