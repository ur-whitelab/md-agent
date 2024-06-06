import mdtraj as md
from langchain.tools import BaseTool

from mdagent.utils import PathRegistry


def load_traj(path_registry, traj_file, top_file=None):
    if top_file is not None:
        traj = md.load(traj_file, top=top_file)
    else:
        traj = md.load(traj_file)
    return traj if traj else None


class Baker_hubbard(BaseTool):
    name = "Baker_hubbard"
    description = """Identify hydrogen bonds based on cutoffs for the Donor-H…Acceptor
    distance and angle."""

    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file, top_file=None):
        traj = load_traj(self.path_registry, traj_file, top_file)
        if not traj:
            return "Trajectory could not be loaded."

        return md.baker_hubbard(
            traj, freq=0.1, exclude_water=True, periodic=True, sidechain_only=False
        )

    async def _arun(self, traj_file, top_file=None):
        raise NotImplementedError("Async version not implemented")


class ComputeKabsch_sander(BaseTool):
    name = "kabsch_sander"
    description = """Compute the Kabsch-Sander hydrogen bond energy between each pair
    of residues in every frame."""

    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file, top_file=None):
        traj = load_traj(self.path_registry, traj_file, top_file)
        if not traj:
            return "Trajectory could not be loaded."
        return md.kabsch_sander(traj)

    async def _arun(self, traj_file, top_file=None):
        raise NotImplementedError("Async version not implemented")


class ComputeWernet_Nilsson(BaseTool):
    name = "wernet_nilsson"
    description = """Identify hydrogen bonds based on cutoffs for the Donor-H…Acceptor
    distance and angle according to the criterion outlined in literature. Angle
    Dependant distance cut off, a "cone" criterion"""

    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file, top_file=None):
        traj = load_traj(self.path_registry, traj_file, top_file)
        if not traj:
            return "Trajectory could not be loaded."
        return md.wernet_nilsson(
            traj, exclude_water=True, periodic=True, sidechain_only=False
        )

    async def _arun(self, traj_file, top_file=None):
        raise NotImplementedError("Async version not implemented")
