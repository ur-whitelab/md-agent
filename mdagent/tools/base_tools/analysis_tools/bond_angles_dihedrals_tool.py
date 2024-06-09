import mdtraj as md
from langchain.tools import BaseTool

from mdagent.utils import PathRegistry, load_single_traj


class ComputeAngles(BaseTool):
    name = "compute_angles"
    description = """Calculate the bond angles for the given sets of three atoms in
    each snapshot, and provide a list of indices specifying which atoms are involved
    in each bond angle calculation.."""

    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file, angle_indices, top_file=None):
        traj = load_single_traj(self.path_registry, traj_file, top_file)
        if not traj:
            return "Trajectory could not be loaded."

        if (
            not angle_indices
            or not isinstance(angle_indices, list)
            or not all(len(indices) == 3 for indices in angle_indices)
        ):
            return (
                "Invalid angle_indices. It should be a list of tuples, each "
                "containing three atom indices."
            )

        return md.compute_angles(traj, angle_indices, periodic=True, opt=True)

    async def _arun(self, traj_file, angle_indices, top_file=None):
        raise NotImplementedError("Async version not implemented")


class ComputeDihedrals(BaseTool):
    name = "compute_dihedrals"
    description = """Calculate the dihedral angles for the given groups of four atoms
      in each snapshot, and provide a list of dihedral angles along with a list of
      indices specifying which atoms are involved in each dihedral angle calculation."""

    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file, indices, top_file=None):
        traj = load_single_traj(self.path_registry, traj_file, top_file)
        if not traj:
            return "Trajectory could not be loaded."

        if (
            not indices
            or not isinstance(indices, list)
            or not all(
                isinstance(tup, tuple) and all(isinstance(i, int) for i in tup)
                for tup in indices
            )
        ):
            return (
                "Invalid indices. It should be a list of tuples, each containing"
                "atom indices as integers."
            )

        # Assuming a generic computation method for demonstration
        # Replace `md.compute_properties` with the actual computation method you need

        return md.compute_dihedrals(traj, indices, periodic=True, opt=True)

    async def _arun(self, traj_file, indices, top_file=None):
        raise NotImplementedError("Async version not implemented")


class ComputePsi(BaseTool):
    name = "compute_psi"
    description = """Calculate the psi angles for each snapshot, providing a list of
    psi angles for each frame in the trajectory and a list of indices specifying the
    atoms involved in calculating each psi angle"""

    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file, top_file=None):
        traj = load_single_traj(self.path_registry, traj_file, top_file)
        if not traj:
            return "Trajectory could not be loaded."
        return md.compute_psi(traj, periodic=True, opt=True)

    async def _arun(self, traj_file, top_file=None):
        raise NotImplementedError("Async version not implemented")


class ComputePhi(BaseTool):
    name = "compute_phi"
    description = """This class calculates phi torsion angles and provides a list of phi
      angles and indices specifying which atoms are involved in the calculations"""

    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file, top_file=None):
        traj = load_single_traj(self.path_registry, traj_file, top_file)
        if not traj:
            return "Trajectory could not be loaded."
        return md.compute_phi(traj, periodic=True, opt=True)

    async def _arun(self, traj_file, top_file=None):
        raise NotImplementedError("Async version not implemented")


class ComputeChi1(BaseTool):
    name = "compute_chi1"
    description = """Calculate the chi1 angles (the first side chain torsion angle
    formed between four atoms around the CA-CB axis) for each snapshot, providing a
    list of chi1 angles and indices specifying the atoms involved in each chi1 angle
    calculation."""

    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file, top_file=None):
        traj = load_single_traj(self.path_registry, traj_file, top_file)
        if not traj:
            return "Trajectory could not be loaded."
        return md.compute_chi1(traj, periodic=True, opt=True)

    async def _arun(self, traj_file, top_file=None):
        raise NotImplementedError("Async version not implemented")


class ComputeChi2(BaseTool):
    name = "compute_chi2"
    description = """Calculate the chi2 angles (the second side chain torsion angle
    formed between four atoms around the CB-CG axis) for each snapshot, providing a
    list of chi2 angles and a list of indices specifying the atoms involved in
    calculating each chi2 angle."""

    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file, top_file=None):
        traj = load_single_traj(self.path_registry, traj_file, top_file)
        if not traj:
            return "Trajectory could not be loaded."
        return md.compute_chi2(traj, periodic=True, opt=True)

    async def _arun(self, traj_file, top_file=None):
        raise NotImplementedError("Async version not implemented")


class ComputeChi3(BaseTool):
    name = "compute_chi3"
    description = """Calculate the chi3 angles (the third side chain torsion angle
    formed between four atoms around the CG-CD axis) for each snapshot in the
    trajectory, providing a list of chi3 angles and indices specifying the atoms
    involved in the calculation of each chi3 angle. Note: Only the residues ARG, GLN,
      GLU, LYS, and MET have these atoms."""

    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file, top_file=None):
        traj = load_single_traj(self.path_registry, traj_file, top_file)
        if not traj:
            return "Trajectory could not be loaded."
        return md.compute_chi3(traj, periodic=True, opt=True)

    async def _arun(self, traj_file, top_file=None):
        raise NotImplementedError("Async version not implemented")


class ComputeChi4(BaseTool):
    name = "compute_chi4"
    description = """Calculate the chi4 angles (the fourth side chain torsion angle
    formed between four atoms around the CD-CE or CD-NE axis) for each snapshot in the
    trajectory, providing a list of indices specifying which atoms are involved in the
    chi4 angle calculations. """

    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file, top_file=None):
        traj = load_single_traj(self.path_registry, traj_file, top_file)
        if not traj:
            return "Trajectory could not be loaded."
        return md.compute_chi4(traj, periodic=True, opt=True)

    async def _arun(self, traj_file, top_file=None):
        raise NotImplementedError("Async version not implemented")


class ComputeOmega(BaseTool):
    name = "compute_omega"
    description = """Calculate the omega angles for each snapshot in the trajectory,
    providing a list of indices specifying which atoms are involved in the omega angle
    calculations.."""

    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file, top_file=None):
        traj = load_single_traj(self.path_registry, traj_file, top_file)
        if not traj:
            return "Trajectory could not be loaded."
        return md.compute_omega(traj, periodic=True, opt=True)

    async def _arun(self, traj_file, top_file=None):
        raise NotImplementedError("Async version not implemented")


# class Ramachandran Plot
