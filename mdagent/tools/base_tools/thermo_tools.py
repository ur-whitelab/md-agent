import mdtraj as md
import numpy as np
from langchain.tools import BaseTool

from mdagent.utils import PathRegistry


def load_traj(path_registry, traj_file, top_file=None):
    if top_file is not None:
        traj = md.load(traj_file, top=top_file)
    else:
        traj = md.load(traj_file)
    return traj if traj else None


class ComputeDipoleMoments(BaseTool):
    name = "ComputeDipoleMoments"
    description = """Compute the total dipole moment for each frame in a
    molecular dynamics trajectory. Requires a trajectory file and
    optionally a topology file and partial charges. Returns an array of
    dipole moments for each frame of the trajectory."""
    # TODO -> should this write to a file or return the array?
    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(
        self, traj_file: str, top_file: str | None = None, charges: np.ndarray = None
    ):
        # todo -> might need to convert inputs to the right types
        traj = load_traj(self.path_registry, traj_file, top_file)
        if not traj:
            return "Failed to load trajectory file."
        if not charges:
            if traj.topology and any(a.element for a in traj.topology.atoms):
                # try to extract charges from the topology
                charges = np.array([a.element.charge for a in traj.topology.atoms])
                if any(charges == None):  # noqa: E711
                    return (
                        "Charges are not available in the topology, "
                        "please provide them manually."
                    )
            else:
                return (
                    "Partial charges must be provided since the "
                    "topology does not contain this information."
                )
        return md.dipole_moments(traj, charges)

    async def _arun(
        self, traj_file: str, top_file: str | None = None, charges: np.ndarray = None
    ):
        raise NotImplementedError("Async version not implemented")


class ComputeStaticDielectric(BaseTool):
    name = "ComputeStaticDielectric"
    description = """Compute the static dielectric constant of a system
      from the dipole moments of a molecular dynamics trajectory.
      Requires a trajectory file, charges, temperature and optionally a
      topology file. Returns the static dielectric
      constant. Returns the static dielectric constant."""
    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(
        self,
        traj_file: str,
        temperature: float,
        top_file: str | None = None,
        charges: np.ndarray = None,
    ):
        # todo -> might need to convert inputs to the right types
        traj = load_traj(self.path_registry, traj_file, top_file)
        if not traj:
            return "Failed to load trajectory file."
        if not charges:
            if traj.topology and any(a.element for a in traj.topology.atoms):
                # try to extract charges from the topology
                charges = np.array([a.element.charge for a in traj.topology.atoms])
                if any(charges == None):  # noqa: E711
                    return (
                        "Charges are not available in the topology, "
                        "please provide them manually."
                    )
            else:
                return (
                    "Partial charges must be provided since the"
                    "topology does not contain this information."
                )
        return str(md.static_dielectric(traj, charges, temperature))

    async def _arun(
        self,
        traj_file: str,
        temperature: float,
        top_file: str | None = None,
        charges: np.ndarray = None,
    ):
        raise NotImplementedError("Async version not implemented")


class ComputeIsothermalCompressabilityKappaT(BaseTool):
    name = "ComputeIsothermalCompressabilityKappaT"
    description = """Compute the isothermal compressibility (kappa_T) for
      a molecular dynamics trajectory. Requires a trajectory file and
      temperature and optionally a topology file. Returns the isothermal
      compressibility."""
    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file: str, temperature: float, top_file: str | None = None):
        # todo -> might need to convert inputs to the right types
        traj = load_traj(self.path_registry, traj_file, top_file)
        if not traj:
            return "Failed to load trajectory file."
        return str(md.isothermal_compressability_kappa_T(traj, temperature))

    async def _arun(
        self, traj_file: str, temperature: float, top_file: str | None = None
    ):
        raise NotImplementedError("Async version not implemented")


class ComputeThermalExpansionAlphaP(BaseTool):
    name = "ComputeThermalExpansionAlphaP"
    description = """Compute the thermal expansion coefficient (alpha_P)
    for a molecular dynamics trajectory. Requires a trajectory file and
      temperature and energies as an array containing the potential
      energies of each trajectory frame, in units of kJ/mol. Optionally
      requires a topology file. Returns the thermal expanssion
      coefficient, units of inverse Kelvin."""
    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file, top_file=None, temperature=300, energies=None):
        # todo -> might need to convert inputs to the right types
        traj = load_traj(self.path_registry, traj_file, top_file)
        if not traj:
            return "Failed to load trajectory file."
        if not energies:
            return (
                "Energies must be provided to compute the thermal"
                "expansion coefficient."
            )
        if energies.shape[0] != traj.n_frames:
            return (
                "The length of the energies array must match the "
                "number of frames in the trajectory."
            )
        return str(md.thermal_expansion_alpha_P(traj, temperature, energies))

    async def _arun(self, traj_file, top_file=None, temperature=300, energies=None):
        raise NotImplementedError("Async version not implemented")


class ComputeDensity(BaseTool):
    name = "ComputeDensity"
    description = """Calculate the mass density of each frame in a
    trajectory. Requires a trajectory file and an array of masses and
    optionally a topology file. Returns an array of mass densities for
    each frame."""
    # todo -> should this write to a file or return the array?
    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file: str, masses: np.ndarray, top_file: str | None = None):
        # todo -> might need to convert inputs to the right types
        traj = load_traj(self.path_registry, traj_file, top_file)
        if not traj:
            return "Failed to load trajectory file."
        if not masses:
            return "Masses must be provided to compute the mass density."
        if masses.shape[0] != traj.n_atoms:
            return (
                "The length of the masses array must match the number"
                "of atoms in the trajectory."
            )
        return md.density(traj, masses)

    async def _arun(
        self, traj_file: str, masses: np.ndarray, top_file: str | None = None
    ):
        raise NotImplementedError("Async version not implemented")
