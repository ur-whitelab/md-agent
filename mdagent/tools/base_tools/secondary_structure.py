import mdtraj as md
from langchain.tools import BaseTool

from mdagent.utils import PathRegistry


def load_traj(path_registry, traj_file, top_file=None):
    if top_file is not None:
        traj = md.load(traj_file, top=top_file)
    else:
        traj = md.load(traj_file)
    return traj if traj else None


class ComputeDSSP(BaseTool):
    name = "ComputeDSSP"
    description = """Compute the DSSP (secondary structure) assignment
    for a protein trajectory. Input is a trajectory file (e.g., .xtc, .
    trr) and an optional topology file (e.g., .pdb, .prmtop). The output
    is an array with the DSSP code for each residue at each time point."""
    # TODO -> should this write to a file or return the array?
    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file, top_file=None):
        traj = load_traj(self.path_registry, traj_file, top_file)
        if not traj:
            return "Trajectory could not be loaded."

        return md.compute_dssp(traj, simplified=True)

    async def _arun(self, traj_file, top_file=None):
        raise NotImplementedError("Async version not implemented")


class ComputeGyrationTensor(BaseTool):
    name = "ComputeGyrationTensor"
    description = """Compute the gyration tensor for each frame in a
    molecular dynamics trajectory. Input is a trajectory file (e.g., .
    xtc, .trr) and an optional topology file (e.g., .pdb, .prmtop).
    The output is an array of gyration tensors for each frame of the
    trajectory."""
    # TODO -> should this write to a file or return the array?
    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file, top_file=None):
        traj = load_traj(self.path_registry, traj_file, top_file)
        if not traj:
            return "Trajectory could not be loaded."
        return md.compute_gyration_tensor(traj)

    async def _arun(self, traj_file, top_file=None):
        raise NotImplementedError("Async version not implemented")


class ComputePrincipleMoments(BaseTool):
    name = "ComputePrincipleMoments"
    description = """Compute the principle moments of inertia for each
      frame in a molecular dynamics trajectory. Input is a trajectory
      file (e.g., .xtc, .trr) and an optional topology file (e.g., .pdb, .
      prmtop). The output is an array of principle moments of inertia for
      each frame of the trajectory."""
    # TODO -> should this write to a file or return the array?
    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file, top_file=None):
        traj = load_traj(self.path_registry, traj_file, top_file)
        if not traj:
            return "Trajectory could not be loaded."
        return md.compute_principal_moments(traj)

    async def _arun(self, traj_file, top_file=None):
        raise NotImplementedError("Async version not implemented")


class ComputeAsphericity(BaseTool):
    name = "ComputeAsphericity"
    description = """Compute the asphericity for each frame in a
    molecular dynamics trajectory. Input is a trajectory file (e.g., .
    xtc, .trr) and an optional topology file (e.g., .pdb, .prmtop).
    The output is an array of asphericity values for each frame of the
    trajectory."""
    # TODO -> should this write to a file or return the array?
    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file, top_file=None):
        traj = load_traj(self.path_registry, traj_file, top_file)
        if not traj:
            return "Trajectory could not be loaded."
        return md.asphericity(traj)

    async def _arun(self, traj_file, top_file=None):
        raise NotImplementedError("Async version not implemented")


class ComputeAcylindricity(BaseTool):
    name = "ComputeAcylindricity"
    description = """Compute the acylindricity for each frame in a
    molecular dynamics trajectory. Input is a trajectory file (e.g., .
    xtc, .trr) and an optional topology file (e.g., .pdb, .prmtop). The
    output is an array of acylindricity values for each frame of the
    trajectory."""
    # TODO -> should this write to a file or return the array?
    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file, top_file=None):
        traj = load_traj(self.path_registry, traj_file, top_file)
        if not traj:
            return "Trajectory could not be loaded."
        return md.acylindricity(traj)

    async def _arun(self, traj_file, top_file=None):
        raise NotImplementedError("Async version not implemented")


class ComputeRelativeShapeAntisotropy(BaseTool):
    name = "ComputeRelativeShapeAntisotropy"
    description = """Compute the relative shape antisotropy for each
    frame in a molecular dynamics trajectory. Input is a trajectory
    file (e.g., .xtc, .trr) and an optional topology file (e.g., .pdb, .
    prmtop). The output is an array of relative shape antisotropy values
    for each frame of the trajectory."""
    # TODO -> should this write to a file or return the array?
    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file, top_file=None):
        traj = load_traj(self.path_registry, traj_file, top_file)
        if not traj:
            return "Trajectory could not be loaded."
        return md.relative_shape_antisotropy(traj)

    async def _arun(self, traj_file, top_file=None):
        raise NotImplementedError("Async version not implemented")
