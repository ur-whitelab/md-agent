import warnings

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import pandas as pd
from langchain.tools import BaseTool

from mdagent.utils import FileType, PathRegistry, load_single_traj, save_plot


class SaltBridgeFunction:
    def __init__(self, path_registry):
        self.path_registry = path_registry
        self.salt_bridge_pairs = []  # stores paired salt bridges
        self.salt_bridge_counts = []
        self.traj = None
        self.traj_file = ""

    def _load_traj(self, traj_file, top_file):
        self.traj = load_single_traj(
            self.path_registry, top_fileid=top_file, traj_fileid=traj_file
        )
        self.traj_file = traj_file if traj_file else top_file

    def find_salt_bridges(
        self,
        threshold_distance: float = 0.4,
        residue_pairs=[],
    ):
        """
        Find Salt Bridge in molecular dynamics simulation trajectory, using
        threshold distance (in nm) between N and O atoms for salt bridge formation,
        based on Barlow and Thornton's original definition of salt bridges
        (https://doi.org/10.1016/S0022-2836(83)80079-5)


        threshold_distance: maximum distance (in nm) between N and O atoms
        residue_pairs (optional): list of tuples (donor_residue, acceptor_residue)
        """
        if self.traj is None:
            raise Exception("MDTrajectory hasn't been loaded")

        if not residue_pairs:
            residue_pairs = [
                # (postive-charged, negative-charged)
                # pairs from https://doi.org/10.1002/prot.22927
                ("ARG", "ASP"),
                ("ARG", "GLU"),
                ("LYS", "ASP"),
                ("LYS", "GLU"),
                ("HIS", "ASP"),
                ("HIS", "GLU"),
            ]
            warnings.warn(
                "No residue pairs provided. Default charged residues "
                "are being used, assuming physiological pH. "
                f"Default pairs: {residue_pairs}",
                UserWarning,
            )

        donor_acceptor_pairs = []
        for pair in residue_pairs:
            print(f"Looking for salt bridges between {pair[0]} and {pair[1]} pairs...")
            donor_atoms = self.traj.topology.select(f'resname == "{pair[0]}"')
            acceptor_atoms = self.traj.topology.select(f'resname == "{pair[1]}"')

            if donor_atoms.size == 0 or acceptor_atoms.size == 0:
                continue

            donor_nitrogens = [  # N atoms in the donor residues (e.g. Arg, Lys, His)
                atom.index
                for atom in self.traj.topology.atoms
                if atom.index in donor_atoms and atom.element.symbol == "N"
            ]
            acceptor_oxygens = [  # O atoms in the acceptor residues (e.g. Asp, Glu)
                atom.index
                for atom in self.traj.topology.atoms
                if atom.index in acceptor_atoms and atom.element.symbol == "O"
            ]

            # generate all possible donor-acceptor pairs
            pairs = np.array(np.meshgrid(donor_nitrogens, acceptor_oxygens)).T.reshape(
                -1, 2
            )
            donor_acceptor_pairs.append(pairs)

        if not donor_acceptor_pairs:
            return None

        donor_acceptor_pairs = np.vstack(donor_acceptor_pairs)  # combine into one list
        all_distances = md.compute_distances(self.traj, donor_acceptor_pairs)

        salt_bridge_counts = []
        salt_bridge_pairs = []
        for frame_idx in range(self.traj.n_frames):
            frame_distances = all_distances[frame_idx]
            within_threshold = frame_distances <= threshold_distance
            salt_bridge_counts.append(np.sum(within_threshold))

            filtered_pairs = donor_acceptor_pairs[within_threshold]
            if filtered_pairs.size > 0:
                salt_bridge_pairs.append((frame_idx, filtered_pairs))
        self.salt_bridge_counts = salt_bridge_counts
        self.salt_bridge_pairs = salt_bridge_pairs

    def plot_salt_bridge_counts(self):
        if not self.salt_bridge_pairs or self.traj.n_frames == 1:
            return None

        plt.figure(figsize=(10, 6))
        plt.plot(
            range(self.traj.n_frames),
            self.salt_bridge_counts,
            marker="o",
            linestyle="-",
            color="b",
        )
        plt.title(f"Salt Bridge Count Over Time - {self.traj_file}")
        plt.xlabel("Frame")
        plt.ylabel("Total Salt Bridge Count")
        plt.grid(True)
        fig_id = save_plot(
            self.path_registry,
            "salt_bridge",
            f"figure of salt bridge counts for {self.traj_file}",
        )
        plt.close()
        return fig_id

    def save_results_to_file(self):
        if self.traj is None:
            raise Exception("Trajectory is None")
        if not self.salt_bridge_pairs:
            return None

        if self.traj.n_frames == 1:
            num_sb = self.salt_bridge_counts[0]
            print(f"We found {num_sb} salt bridges for {self.traj_file}.")
            print(
                (
                    "Since the trajectory has only one frame, we saved a "
                    "list of salt bridges instead of plotting."
                )
            )

            salt_bridge_data = []
            frame_idx, bridges = self.salt_bridge_pairs[0]
            for bridge in bridges:
                donor_residue = self.traj.topology.atom(bridge[0]).residue
                acceptor_residue = self.traj.topology.atom(bridge[1]).residue
                salt_bridge_data.append(
                    {
                        "Donor": f"{donor_residue.name} ({donor_residue.index + 1})",
                        "Acceptor": f"{acceptor_residue.name} ({acceptor_residue.index + 1})",
                    }
                )
            df = pd.DataFrame(salt_bridge_data)

        else:
            df = pd.DataFrame(
                {
                    "Frame": range(self.traj.n_frames),
                    "Salt Bridge Count": self.salt_bridge_counts,
                }
            )

        # save to file, add to path registry
        file_name = self.path_registry.write_file_name(
            FileType.RECORD,
            record_type="salt_bridges",
            file_format="csv",
        )
        file_id = self.path_registry.get_fileid(file_name, FileType.RECORD)
        file_path = f"{self.path_registry.ckpt_records}/{file_name}"
        df.to_csv(file_path, index=False)
        self.path_registry.map_path(
            file_id, file_path, description=f"salt bridge analysis for {self.traj_file}"
        )
        return file_id

    def compute_salt_bridges(
        self,
        traj_file,
        top_file,
        threshold_distance,
        residue_pairs,
    ):
        self._load_traj(traj_file, top_file)
        self.find_salt_bridges(threshold_distance, residue_pairs)
        file_id = self.save_results_to_file()
        fig_id = self.plot_salt_bridge_counts()
        return file_id, fig_id


class SaltBridgeTool(BaseTool):
    name = "SaltBridgeTool"
    description = (
        "A tool to find and count salt bridges in a protein trajectory. "
        "You need to provide either PDB file or trajectory and topology files. "
        "Optional: provide threshold distance (default:0.4) and a custom list "
        "of residue pairs as tuples of positive-charged and negative-charged. "
    )
    path_registry: PathRegistry | None = None

    def __init__(self, path_registry):
        super().__init__()
        self.path_registry = path_registry

    def _run(
        self,
        traj_file: str,
        top_file: str | None = None,
        threshold_distance=0.4,
        residue_pairs=[],
    ):
        try:
            if self.path_registry is None:
                return "Path registry is not set"

            salt_bridge_function = SaltBridgeFunction(self.path_registry)
            results_file_id, fig_id = salt_bridge_function.compute_salt_bridges(
                traj_file, top_file, threshold_distance, residue_pairs
            )
            if not results_file_id:
                return (
                    "Succeeded. No salt bridges are found in "
                    f"{salt_bridge_function.traj_file}."
                )

            message = f"Saved results with file id: {results_file_id} "
            if fig_id:
                message += f"and figure with fig id {fig_id}."
            return "Succeeded. " + message
        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"
