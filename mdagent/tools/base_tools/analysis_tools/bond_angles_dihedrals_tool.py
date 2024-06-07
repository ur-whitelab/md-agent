import mdtraj as md
from langchain.tools import BaseTool

from mdagent.utils import PathRegistry, load_single_traj

# def load_single_traj(path_registry, traj_file, top_file=None):
#     if top_file is not None:
#         traj = md.load(traj_file, top=top_file)
#     else:
#         traj = md.load(traj_file)
#     return traj if traj else None


class ComputeAngles(BaseTool):
    name = "compute_angles"
    description = """Calculate the bond angles for the given sets of three atoms in
    each snapshot of a molecular simulation."""

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
    in each snapshot of a molecular simulation."""

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
    description = """Calculate the psi angles for each snapshot in a molecular dynamics
      simulation. These angles involve the alpha and carbonyl carbons, allowing rotation
        that shapes the protein's secondary structure and overall 3D conformation.."""

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
    description = """This class is designed to calculate the phi torsion angles in a
    molecular dynamics simulation.
    Involve the nitrogen and alpha carbon, allowing for rotation that contributes to
    the overall folding pattern
     More flexible, allowing for a wide range of conformations that define the protein’s
     three-dimensional structure ."""

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
    formed between four atoms around the CA-CB axis) for each snapshot in a molecular
      dynamics simulation."""

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


class Compute_Chi2(BaseTool):
    name = "compute_chi2"
    description = """Calculate the chi2 angles (the second side chain torsion angle
      formed between four atoms around the CB-CG axis) for each snapshot in a
      molecular dynamics simulation."""

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
    formed between four atoms around the CG-CD axis) for each snapshot in a molecular
    dynamics simulation. Note: Only the residues ARG, GLN, GLU, LYS, and MET have
    these atoms."""

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
      formed between four atoms around the CD-CE or CD-NE axis) for each snapshot in
      a molecular dynamics simulation. Note: Only the residues ARG and LYS have these
        atoms."""

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
    description = """Calculate the omega angles (a specific type of bond angle) for
    each snapshot in a molecular dynamics simulation.
    Role: omega angles are primarily involved in the peptide bond, crucial for
    determining the planarity
      and rigidity of the peptide bond. They usually have less variability due to
      the preference for trans configuration.
     omega angles are more rigid due to the partial double-bond character of the
     peptide bond, leading to limited rotational freedom
     (mostly fixed at 180° or 0°)."""

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
