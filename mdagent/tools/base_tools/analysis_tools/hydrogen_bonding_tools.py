import mdtraj as md
from langchain.tools import BaseTool

from mdagent.utils import PathRegistry, load_single_traj

# def load_single_traj(path_registry, traj_file, top_file=None):
#     if top_file is not None:
#         traj = md.load(traj_file, top=top_file)
#     else:
#         traj = md.load(traj_file)
#     return traj if traj else None


class BakerHubbard(BaseTool):
    name = "Baker_hubbard"
    description = """Identify hydrogen bonds based that are present in at least 10%
     of each frames ( freq =0.1) and provides a list of tuples with each tuples
    containing three  integers representing the indices of atoms (donor, hydrogen,
    acceptor) involved in the hydrogen bonding."""

    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file, top_file=None, freq=0.1):
        traj = load_single_traj(self.path_registry, top_file, traj_file)
        if not traj:
            return "Trajectory could not be loaded."

        return md.baker_hubbard(
            traj, freq, exclude_water=True, periodic=True, sidechain_only=False
        )

    async def _arun(self, traj_file, top_file=None, freq=0.1):
        raise NotImplementedError("Async version not implemented")


class KabschSander(BaseTool):
    name = "kabsch_sander"
    description = """Compute hydrogen bond energy between each pair
    of residues in every frame of the simulation and provides list of indices
    specifying which residues are involved in each hydrogen bond and its hydrogen bond
    energies."""

    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file, top_file=None):
        traj = load_single_traj(self.path_registry, top_file, traj_file)
        if not traj:
            return "Trajectory could not be loaded."
        return md.kabsch_sander(traj)

    async def _arun(self, traj_file, top_file=None):
        raise NotImplementedError("Async version not implemented")


class WernetNilsson(BaseTool):
    name = "wernet_nilsson"
    description = """Identifies hydrogen bonds without frequency parameter, provides
    a list of tuples with indices of donor, hydrogen and acceptor atoms."""

    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file, top_file=None):
        traj = load_single_traj(self.path_registry, top_file, traj_file)
        if not traj:
            return "Trajectory could not be loaded."
        return md.wernet_nilsson(
            traj, exclude_water=True, periodic=True, sidechain_only=False
        )

    async def _arun(self, traj_file, top_file=None):
        raise NotImplementedError("Async version not implemented")


# 3d visualization?
# heatmap?
# time series plots
# histograms
