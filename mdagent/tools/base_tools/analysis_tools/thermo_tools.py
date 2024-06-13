import csv
import json

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
from langchain.tools import BaseTool

from mdagent.utils import FileType, PathRegistry, load_single_traj


def load_charge_json(charge_file: str):
    with open(charge_file, "r") as f:
        charges = json.load(f)
    return charges


def match_charges_to_traj(atom_charges: dict, traj: md.Trajectory):
    topology = traj.topology

    charges = []
    for atom in topology.atoms:
        atom_name = atom.name
        if atom_name in atom_charges:
            charges.append(atom_charges[atom_name])
        else:
            raise ValueError(f"Charge for atom '{atom_name}' not found in JSON file")
        if len(charges) != len(traj.xyz[0]):
            raise ValueError(
                "Number of charges does not match number of atoms in trajectory"
            )
    return np.array(charges)


def get_charges(path_registry, traj: md.Trajectory, charge_file_id: str):
    try:
        charge_json = path_registry.get_mapped_path(charge_file_id)
    except Exception:
        raise FileNotFoundError(
            "Failed to load charge file. File ID not found in path registry."
        )
    try:
        charges = load_charge_json(path_registry, charge_json)
    except Exception:
        raise Exception(
            "Failed to load charge file. Charge file should be a "
            "json file mapping each atom in the trajectory to its partial charge."
        )
    try:
        charges = match_charges_to_traj(charges, traj)
    except Exception:
        raise Exception(
            "Failed to match charges to trajectory. Please ensure that "
            "Each atom in the trajectory is mapped to its partial charge in the json file."
        )


class ComputeDipoleMoments(BaseTool):
    name = "ComputeDipoleMoments"
    description = """Compute the total dipole moment for each frame in a
    molecular dynamics trajectory. Requires a trajectory file and a json file mapping
    each atom in the trajectory to its partial charge. Optionally requires a topology file.
    Returns an array of dipole moments for each frame of the trajectory, written to a file."""
    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry
        if not self.path_registry:
            self.path_registry = PathRegistry().get_instance()

    def _run(self, traj_file: str, charge_file: str, top_file: str | None = None):
        # load traj
        traj = load_single_traj(
            path_registry=self.path_registry, traj_fileid=traj_file, top_fileid=top_file
        )
        if not traj:
            return "Failed to load trajectory file."

        # load charges
        try:
            charges = get_charges(traj, charge_file)
        except Exception as e:
            return str(e)

        # compute dipole moments
        try:
            dipole_moments = md.dipole_moments(traj, charges)
        except Exception as e:
            return str(e)

        # save dipole moments to file
        file_name = self.path_registry.write_file_name(
            type=FileType.UNKNOWN,
            fig_analysis=f"dipole_moments_{traj_file}",
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
        return "Dipole moments computed and saved to file {file_id}."

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
      constant."""
    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry
        if not self.path_registry:
            self.path_registry = PathRegistry().get_instance()

    def _run(
        self,
        traj_file: str,
        temperature: str,
        charge_file: str,
        top_file: str | None = None,
    ):
        # convert temperature to float
        try:
            temperature = float(temperature)
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
            charges = get_charges(traj, charge_file)
        except Exception as e:
            return str(e)

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
        if not self.path_registry:
            self.path_registry = PathRegistry().get_instance()

    def _run(self, traj_file: str, temperature: str, top_file: str | None = None):
        # convert temperature to float
        try:
            temperature = float(temperature)
        except Exception:
            return "Temperature must be a number."
        traj = load_single_traj(
            path_registry=self.path_registry, traj_fileid=traj_file, top_fileid=top_file
        )
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
        if not self.path_registry:
            self.path_registry = PathRegistry().get_instance()

    def _load_energies(self, energy_file: str):
        if energy_file.endswith(".csv"):
            csv_reader = csv.DictReader(energy_file)
            potential_energy_column = None
            for field in csv_reader.fieldnames:
                if "potential" in field.lower() and "energy" in field.lower():
                    potential_energy_column = field
                    break
            if potential_energy_column is None:
                raise ValueError(
                    "CSV file must contain a column with both 'potential' and 'energy' in its name."
                )
            for row in csv_reader:
                potential_energies.append(float(row[potential_energy_column]))
        elif energy_file.endswith(".json"):
            with open(energy_file, "r") as f:
                data = json.load(f)
            potential_energy_key = None
            for key in data:
                if "potential" in key.lower() and "energy" in key.lower():
                    potential_energy_key = key
                    break
            if potential_energy_key is None:
                raise ValueError(
                    "JSON file must contain a key with both 'potential' and 'energy' in its name."
                )
            potential_energies = data[potential_energy_key]
        else:
            raise ValueError(
                "Unsupported file format. Please upload a CSV or JSON file."
            )
        return potential_energies

    def _run(
        self,
        traj_file,
        temperature: str,
        energy_file: str,
        top_file=None,
    ):
        # convert temperature to float
        try:
            temperature = float(temperature)
        except Exception:
            return "Temperature must be a number."

        traj = load_single_traj(
            path_registry=self.path_registry, traj_fileid=traj_file, top_fileid=top_file
        )
        if not traj:
            return "Failed to load trajectory file."

        try:
            energies = self._load_energies(energy_file)
        except Exception as e:
            return str(e)
        if not energies:
            return (
                "Error loading energies file. Please ensure that the file "
                "contains a list of potential energies for each frame."
            )
        if energies.shape[0] != traj.n_frames:
            return (
                "The length of the energies array must match the number"
                "of frames in the trajectory."
            )
        return str(md.thermal_expansion_alpha_P(traj, temperature, energies))

    async def _arun(self, traj_file, top_file=None, temperature=300, energies=None):
        raise NotImplementedError("Async version not implemented")


class ComputeMassDensity(BaseTool):
    name = "ComputeMassDensity"
    description = """Calculate the mass density of each frame in a
    trajectory. Requires a trajectory file optionally
    a topology file. Returns an array of mass densities for
    each frame."""
    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry
        if not self.path_registry:
            self.path_registry = PathRegistry().get_instance()

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
        return None

    def _run(self, traj_file: str, top_file: str | None = None):
        # todo -> might need to convert inputs to the right types
        traj = load_single_traj(
            path_registry=self.path_registry, traj_fileid=traj_file, top_fileid=top_file
        )
        if not traj:
            return "Failed to load trajectory file."
        # todo -> load specific masses file provided by user
        # low priority, as usually masses are standard

        mass_density = md.density(traj)
        # save masses to file
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

        frame_ids = np.arange(len(mass_density))
        data = np.column_stack((frame_ids, mass_density))

        np.savetxt(
            save_path,
            data,
            delimiter=",",
            header="Frame ID,Density (g/cm^3)",
            fmt="%d,%.6f",
            comments="",
        )
        try:
            self._plot_data(data, traj_file)
        except Exception as e:
            return f"Mass density computed and saved to file {file_id}. Failed to plot data: {str(e)}"
        return f"Mass density computed and saved to file {file_id}."

    async def _arun(
        self, traj_file: str, masses: np.ndarray, top_file: str | None = None
    ):
        raise NotImplementedError("Async version not implemented")
