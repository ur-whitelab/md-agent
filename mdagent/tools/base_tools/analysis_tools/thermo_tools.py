import json

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import openmm
from langchain.tools import BaseTool
from openmm import app

from mdagent.utils import FileType, PathRegistry, load_single_traj


class GetTrajCharges:
    def compute_charges_from_traj(
        self, traj: md.Trajectory, forcefield: str = "amber14-all.xml"
    ) -> np.ndarray:
        """
        This function computes the partial charges of each atom in a
            trajectory using OpenMM.
        The charges are computed using the forcefield specified
            in the forcefield argument.
        The charges are returned as a numpy array.

        Parameters:
        traj: The trajectory for which charges are to be computed.
        forcefield: The forcefield to be used for computing charges.
            Default is 'amber14-all.xml'.

        Returns:
        charges: A numpy array containing the partial charges of each
            atom in the trajectory.
        """
        forcefield_loaded = app.ForceField(forcefield)
        modeller = app.Modeller(traj.top.to_openmm(), traj.openmm_positions(0))
        system = forcefield_loaded.createSystem(modeller.topology)
        charges = np.zeros(traj.n_atoms)
        for force in system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                for i in range(force.getNumParticles()):
                    charge, _, _ = force.getParticleParameters(i)
                    charges[i] = charge._value
        return charges

    def _load_charge_json(self, charge_file: str) -> dict:
        with open(charge_file, "r") as f:
            charges = json.load(f)
        return charges

    def match_charges_to_traj(
        self, atom_charges: dict, traj: md.Trajectory
    ) -> np.ndarray:
        topology = traj.topology

        charges = []
        for atom in topology.atoms:
            atom_name = atom.name
            if atom_name in atom_charges:
                charges.append(atom_charges[atom_name])
            else:
                raise ValueError(
                    f"Charge for atom '{atom_name}' not found in JSON file"
                )
        if len(charges) != len(traj.xyz[0]):
            raise ValueError(
                "Number of charges does not match number of atoms in trajectory"
            )
        return np.array(charges)

    def get_charges_given_file(
        self, path_registry, traj: md.Trajectory, charge_file_id: str
    ) -> np.ndarray:
        try:
            charge_json = path_registry.get_mapped_path(charge_file_id)
        except Exception:
            raise FileNotFoundError(
                "Failed to load charge file. File ID not found in path registry."
            )
        try:
            charges = self._load_charge_json(charge_json)
        except Exception:
            raise Exception(
                "Failed to load charge file. Charge file should be a "
                "json file mapping each atom in the trajectory to its partial charge."
            )
        try:
            charges = self.match_charges_to_traj(charges, traj)
        except Exception:
            raise Exception(
                "Failed to match charges to trajectory. Please ensure that "
                "Each atom in the trajectory is mapped to its partial charge "
                "in the json file."
            )

    def get_charges(
        self, path_registry, traj: md.Trajectory, charge_file_id: str | None = None
    ) -> np.ndarray:
        if charge_file_id:
            try:
                return self.get_charges_given_file(path_registry, traj, charge_file_id)
            except Exception:
                pass
        return self.compute_charges_from_traj(traj)


class ComputeDipoleMoments(BaseTool):
    name = "ComputeDipoleMoments"
    description = """Compute the total dipole moment for each frame in a
    molecular dynamics trajectory. Requires a trajectory file ID and optionally
    a topology file ID. If the user provided a charges file, you should
    provide the file ID for the charges file also. Otherwise, the charges
    will be computed from the trajectory using OpenMM.
    Returns an array of dipole moments for each frame of the trajectory,
    written to a file."""
    path_registry: PathRegistry = PathRegistry().get_instance()
    get_charges = GetTrajCharges()

    def __init__(self, path_registry: PathRegistry | None = None):
        super().__init__()
        if path_registry is not None:
            self.path_registry = path_registry
        self.get_charges = GetTrajCharges()

    def _compute_dipole_moments(self, traj: md.Trajectory, charges: np.ndarray):
        return md.dipole_moments(traj, charges)

    def _save_dipoles_to_file(self, dipole_moments: np.ndarray, traj_file: str):
        file_name = self.path_registry.write_file_name(
            type=FileType.RECORD,
            record_type="dipole_moments",
            file_format="csv",
        )

        save_path = f"{self.path_registry.ckpt_files}/{file_name}"
        description = (
            f"Dipole moments for each frame in the trajectory for file {traj_file}"
        )

        file_id = self.path_registry.get_fileid(
            file_name=file_name, type=FileType.UNKNOWN
        )
        self.path_registry.map_path(file_id, save_path, description)

        np.savetxt(
            save_path,
            dipole_moments,
            delimiter=",",
            header="Dipole_X,Dipole_Y,Dipole_Z",
            comments="",
        )
        return file_id

    def _run(
        self,
        traj_file: str,
        charge_file: str | None = None,
        top_file: str | None = None,
    ):
        try:
            traj = load_single_traj(
                path_registry=self.path_registry,
                traj_fileid=traj_file,
                top_fileid=top_file,
            )
            if not traj:
                raise Exception("Failed to load trajectory file.")
        except Exception as e:
            return str(e)

        try:
            charges = self.get_charges.get_charges(
                self.path_registry, traj, charge_file
            )
        except Exception as e:
            return str(e)

        dipole_moments = self._compute_dipole_moments(traj, charges)
        file_id = self._save_dipoles_to_file(dipole_moments, traj_file)

        return "Dipole moments computed and " f"saved to file {file_id}."

    async def _arun(
        self, traj_file: str, top_file: str | None = None, charges: np.ndarray = None
    ):
        raise NotImplementedError("Async version not implemented")


class ComputeStaticDielectric(BaseTool):
    name = "ComputeStaticDielectric"
    description = """Compute the static dielectric constant of a system
      from the dipole moments of a molecular dynamics trajectory.
      Requires a trajectory file, temperature and optionally a
      topology file. If the user provided a charges file, you should
        provide the file ID for the charges file also. Otherwise, the charges
        will be computed from the trajectory using OpenMM.
      Returns the static dielectric
      constant."""
    path_registry: PathRegistry = PathRegistry().get_instance()
    get_charges = GetTrajCharges()

    def __init__(self, path_registry: PathRegistry | None = None):
        super().__init__()
        if path_registry is not None:
            self.path_registry = path_registry
        self.get_charges = GetTrajCharges()

    def _compute_static_dielectric(
        self, traj: md.Trajectory, charges: np.ndarray, temperature: float
    ):
        return md.static_dielectric(traj, charges, temperature)

    def _run(
        self,
        traj_file: str,
        temperature: str,
        charge_file: str | None = None,
        top_file: str | None = None,
    ):
        # convert temperature to float
        try:
            temp_ = float(temperature)
        except Exception:
            return "Temperature must be a number."

        # load traj
        traj = load_single_traj(
            path_registry=self.path_registry, traj_fileid=traj_file, top_fileid=top_file
        )
        if not traj:
            return "Failed to load trajectory file."

        # load charges
        try:
            charges = self.get_charges.get_charges(
                self.path_registry, traj, charge_file
            )
        except Exception as e:
            return str(e)

        return str(self._compute_static_dielectric(traj, charges, temp_))

    async def _arun(
        self,
        traj_file: str,
        temperature: str,
        charge_file: str,
        top_file: str | None = None,
    ):
        raise NotImplementedError("Async version not implemented")


class ComputeIsothermalCompressabilityKappaT(BaseTool):
    name = "ComputeIsothermalCompressabilityKappaT"
    description = """Compute the isothermal compressibility (kappa_T) for
    a molecular dynamics trajectory. Requires a trajectory file and
    temperature. Returns the isothermal
    compressibility."""
    path_registry: PathRegistry = PathRegistry().get_instance()

    def __init__(self, path_registry: PathRegistry | None = None):
        super().__init__()
        if path_registry is not None:
            self.path_registry = path_registry

    def _compute_isothermal_compressability_kappa_T(
        self, traj: md.Trajectory, temperature: float
    ):
        return md.isothermal_compressability_kappa_T(traj, temperature)

    def _run(self, traj_file: str, temperature: str, top_file: str | None = None):
        # convert temperature to float
        try:
            temp_ = float(temperature)
        except Exception:
            return "Temperature must be a number."
        traj = load_single_traj(
            path_registry=self.path_registry, traj_fileid=traj_file, top_fileid=top_file
        )
        if not traj:
            return "Failed to load trajectory file."
        return str(self._compute_isothermal_compressability_kappa_T(traj, temp_))

    async def _arun(
        self, traj_file: str, temperature: float, top_file: str | None = None
    ):
        raise NotImplementedError("Async version not implemented")


class ComputeMassDensity(BaseTool):
    name = "ComputeMassDensity"
    description = """Calculate the mass density of each frame in a
    trajectory. Requires a trajectory file optionally
    a topology file. Returns an array of mass densities for
    each frame."""
    path_registry: PathRegistry = PathRegistry().get_instance()

    def __init__(self, path_registry: PathRegistry | None = None):
        super().__init__()
        if path_registry is not None:
            self.path_registry = path_registry

    def _plot_data(self, data: np.ndarray, traj_file: str):
        plt.figure(figsize=(10, 6))
        plt.plot(
            data[:, 0],
            data[:, 1],
            label="Mass Density",
            color="blue",
            marker="o",
            linestyle="-",
        )
        plt.title("Mass Density Over Simulation")
        plt.xlabel("Frame")
        plt.ylabel("Density (g/cm³)")
        plt.grid(True)
        plt.legend()

        fig_name = self.path_registry.write_file_name(
            type=FileType.FIGURE,
            fig_analysis=f"mass_density_{traj_file}",
            file_format="png",
        )

        plot_path = f"{self.path_registry.ckpt_figures}/{fig_name}"
        description = (
            f"Mass density for each frame in the trajectory for file {traj_file}"
        )

        file_id = self.path_registry.get_fileid(
            file_name=fig_name, type=FileType.FIGURE
        )
        self.path_registry.map_path(file_id, plot_path, description)
        plt.savefig(plot_path)
        return file_id

    def _compute_density(self, traj: md.Trajectory) -> np.ndarray:
        mass_density = md.density(traj)
        return mass_density

    def _stack_data(self, mass_density: np.ndarray) -> np.ndarray:
        frame_ids = np.arange(len(mass_density))
        data = np.column_stack((frame_ids, mass_density))
        return data

    def stack_and_save(self, data: np.ndarray, traj_file: str) -> str:
        file_name = self.path_registry.write_file_name(
            type=FileType.UNKNOWN,
            fig_analysis=f"mass_density_{traj_file}",
            file_format="csv",
        )

        save_path = f"{self.path_registry.ckpt_files}/{file_name}"
        description = (
            f"Mass density for each frame in the trajectory for file {traj_file}"
        )

        file_id = self.path_registry.get_fileid(
            file_name=file_name, type=FileType.UNKNOWN
        )
        self.path_registry.map_path(file_id, save_path, description)

        np.savetxt(
            save_path,
            data,
            delimiter=",",
            header="Frame ID,Density (g/cm^3)",
            fmt="%d,%.6f",
            comments="",
        )
        return file_id

    def _run(self, traj_file: str, top_file: str | None = None):
        traj = load_single_traj(
            path_registry=self.path_registry, traj_fileid=traj_file, top_fileid=top_file
        )
        if not traj:
            return "Failed to load trajectory file."
        # todo -> load specific masses file provided by user
        # low priority, as usually masses are standard

        mass_density = md.density(traj)
        data = self._stack_data(mass_density)
        file_id = self.stack_and_save(data, traj_file)
        try:
            plot_id = self._plot_data(data, traj_file)
        except Exception as e:
            return (
                "Mass density computed and saved "
                f"to file {file_id}. "
                f"Failed to plot data: {str(e)}"
            )
        return (
            f"Mass density computed and saved to file {file_id}."
            f"Plot saved to file {plot_id}."
        )

    async def _arun(
        self, traj_file: str, masses: np.ndarray, top_file: str | None = None
    ):
        raise NotImplementedError("Async version not implemented")
