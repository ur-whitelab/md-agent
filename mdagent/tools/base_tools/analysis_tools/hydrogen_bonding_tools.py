import json
import os

import mdtraj as md
from langchain.tools import BaseTool

from mdagent.utils import PathRegistry, load_single_traj


class HydrogenBondTool(BaseTool):
    name = "hydrogen_bond_tool"
    description = """Base class for hydrogen bond analysis tools."""

    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file, top_file=None):
        raise NotImplementedError("Subclasses should implement this method.")

    async def _arun(self, traj_file, top_file=None):
        raise NotImplementedError("Async version not implemented")

    def save_results_to_file(self, results, file_name):
        with open(file_name, "w") as f:
            json.dump(results, f)


class BakerHubbard(HydrogenBondTool):
    name = "baker_hubbard"
    description = """Identify hydrogen bonds that are present in at least 10% of each
    frame (freq=0.1). Provides a list of tuples with each tuple containing three
    integers representing the indices of atoms (donor, hydrogen, acceptor) involved in
    the hydrogen bonding."""

    def __init__(
        self,
        path_registry: PathRegistry,
        exclude_water=True,
        periodic=True,
        sidechain_only=False,
    ):
        super().__init__(path_registry)
        self.exclude_water = exclude_water
        self.periodic = periodic
        self.sidechain_only = sidechain_only

    def _run(self, traj_file, top_file=None, freq=0.1):
        try:
            if not top_file:
                top_file = self.top_file(traj_file)

            traj = load_single_traj(self.path_registry, top_file, traj_file)
            if not traj:
                return """Failed. Trajectory could not be loaded; unable to retrieve
                data needed to find hydrogen bonds. This may be due missing files,
                corrupted files, or incorrect formatted file. Please check and try
                again."""

            result = md.baker_hubbard(
                traj,
                freq,
                exclude_water=self.exclude_water,
                periodic=self.periodic,
                sidechain_only=self.sidechain_only,
            )
            return f"Succeeded. {result}"

        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"

    def top_file(self, traj_file):
        top_file = os.path.join(os.path.dirname(traj_file), "topology.pdb")
        return top_file


class KabschSander(HydrogenBondTool):
    name = "kabsch_sander"
    description = """Calculate the energy of hydrogen bonds between pairs of
    residues in each frame of the simulation. It shows which residues are
    forming hydrogen bonds and the energy of these bonds for
      each frame."."""

    def _run(self, traj_file, top_file=None):
        try:
            if not top_file:
                top_file = self.top_file(traj_file)
            traj = load_single_traj(self.path_registry, top_file, traj_file)
            if not traj:
                return """Failed. Trajectory could not be loaded; unable to access
            data required to calculate hydrogen bond energies. This could be due to
            missing files, corrupted files, or incorrect formatted file. Please check
            and try again."""

            result = md.kabsch_sander(traj)
            return f"Succeeded. {result}"

        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"

    def top_file(self, traj_file):
        top_file = os.path.join(os.path.dirname(traj_file), "topology.pdb")
        return top_file


class WernetNilsson(HydrogenBondTool):
    name = "wernet_nilsson"
    description = """Identifies hydrogen bonds without a frequency parameter. Provides
    a list of tuples with indices of donor, hydrogen, and acceptor atoms. Prefer this
    tool over BakerHubbard unless explicitly requested."""

    def __init__(
        self,
        path_registry: PathRegistry,
        exclude_water=True,
        periodic=True,
        sidechain_only=False,
    ):
        super().__init__(path_registry)
        self.exclude_water = exclude_water
        self.periodic = periodic
        self.sidechain_only = sidechain_only

    def _run(self, traj_file, top_file=None):
        try:
            if not top_file:
                top_file = self.top_file(traj_file)
            traj = load_single_traj(self.path_registry, top_file, traj_file)
            if not traj:
                return """Failed. Trajectory could not be loaded' unable to retrieve
                data needed to find hydrogen bonds. This may be due missing files,
                corrupted files, or incorrect formatted file. Please check and try
                again"""

            result = md.wernet_nilsson(
                traj,
                exclude_water=self.exclude_water,
                periodic=self.periodic,
                sidechain_only=self.sidechain_only,
            )
            return f"Succeeded. {result}"

        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"

    def top_file(self, traj_file):
        top_file = os.path.join(os.path.dirname(traj_file), "topology.pdb")
        return top_file


# def main():
#     # Check if user mentioned "BakerHubbard"
#     if "bakerhubbard" in user_input.lower():
#         tool = BakerHubbard(path_registry=path_instance)
#     else:
#         tool = WernetNilsson(path_registry=path_instance)  # Default tool

#     result = tool._run(traj_file, top_file)

#  print(result)


# if __name__ == "__main__":
#  main()
