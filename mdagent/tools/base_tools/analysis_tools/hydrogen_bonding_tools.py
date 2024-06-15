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


class BakerHubbard(HydrogenBondTool):
    name = "baker_hubbard"
    description = """Identify hydrogen bonds that are present in at least 10% of each
    frame (freq=0.1). Provides a list of tuples with each tuple containing three
    integers representing the indices of atoms (donor, hydrogen, acceptor) involved in
    the hydrogen bonding."""

    def _run(self, traj_file, top_file=None, freq=0.1):
        try:
            traj = load_single_traj(self.path_registry, top_file, traj_file)
            if not traj:
                return "Failed. Trajectory could not be loaded."

            result = md.baker_hubbard(
                traj, freq, exclude_water=True, periodic=True, sidechain_only=False
            )
            return f"Succeeded. {result}"

        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"


class KabschSander(HydrogenBondTool):
    name = "kabsch_sander"
    description = """Compute hydrogen bond energy between each pair of residues in
    every frame of the simulation. Provides a list of indices specifying which residues
    are involved in each hydrogen bond and its hydrogen bond energies."""

    def _run(self, traj_file, top_file=None):
        try:
            traj = load_single_traj(self.path_registry, top_file, traj_file)
            if not traj:
                return "Failed. Trajectory could not be loaded."

            result = md.kabsch_sander(traj)
            return f"Succeeded. {result}"

        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"


class WernetNilsson(HydrogenBondTool):
    name = "wernet_nilsson"
    description = """Identifies hydrogen bonds without a frequency parameter. Provides
    a list of tuples with indices of donor, hydrogen, and acceptor atoms. Prefer this
    tool over BakerHubbard unless explicitly requested."""

    def _run(self, traj_file, top_file=None):
        try:
            traj = load_single_traj(self.path_registry, top_file, traj_file)
            if not traj:
                return "Failed. Trajectory could not be loaded."

            result = md.wernet_nilsson(
                traj, exclude_water=True, periodic=True, sidechain_only=False
            )
            return f"Succeeded. {result}"

        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"


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
